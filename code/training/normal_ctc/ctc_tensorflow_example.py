import time
import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np

from six.moves import xrange as range
from python_speech_features import mfcc

from feature_extraction import extraction
from data_handler.loaders import tf_loader
from .utils import maybe_download as maybe_download
from .utils import sparse_tuple_from as sparse_tuple_from

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

# Some configs
num_features = 20
# Accounting the 0th indice +  space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parameters
num_epochs = 1000
num_hidden = 250
num_layers = 3
batch_size = 2
initial_learning_rate = 1e-2
momentum = 0.9

num_examples = 10
num_batches_per_epoch = int(num_examples/batch_size)

# Loading the data

data = tf_loader.load_data_from_file('../data/umeerj/data.dat', num_features, num_examples)
sampless = [d[1] for d in data]
#samples = tf_loader.pad_data2(samples, 0)
originals = [d[2] for d in data]
data = None

samples_min = np.min([np.min(s) for s in sampless])
samples_max = np.max([np.max(s) for s in sampless])

#sampless = (samples - np.mean(samples)) / np.std(samples)

targets = [' '.join(t.strip().lower().split(' '))
               .replace('.', '').replace('-', '').replace("'", '').replace(':', '').replace(',', '').replace('?', '').replace('!', '')
           for t in originals]
targets = [target.replace(' ', '  ') for target in targets]
targets = [target.split(' ') for target in targets]
targets = [np.hstack([SPACE_TOKEN if x == '' else list(x) for x in target]) for target in targets]
targetss = [np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in target]) for target in targets]

#for samples1, targets, samples_len, _ in batch_generator:
#    targets1 = sparse_tuple_from(targets)

# THE MAIN CODE!

graph = tf.Graph()
with graph.as_default():
    # e.g: log filter bank or MFCC features
    # Has size [batch_size, max_stepsize, num_features], but the
    # batch_size and max_stepsize can vary along each step
    inputs = tf.placeholder(tf.float32, [None, None, num_features], name='inputs')

    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32, name='spars')

    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    def lstm_cell():
        return tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

    stack = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell() for _ in range(num_layers)])

    # The second output is the last state and we will no use that
    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]

    # Reshaping to apply the same weights over the timesteps
    outputs = tf.reshape(outputs, [-1, num_hidden])

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    W = tf.Variable(tf.truncated_normal([num_hidden,
                                         num_classes],
                                        stddev=0.1))
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    b = tf.Variable(tf.constant(0., shape=[num_classes]))

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.MomentumOptimizer(initial_learning_rate,
                                           0.9).minimize(cost)

    # Option 2: tf.nn.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

    # Inaccuracy: label error rate
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          targets))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(graph=graph, config=config) as session:
    # Initializate the weights and biases
    tf.global_variables_initializer().run()

    for curr_epoch in range(num_epochs):
        train_cost = train_ler = 0
        start = time.time()

        for train_inputs, train_targets, train_seq_len, _ in \
                tf_loader.batch_generator(sampless, targetss, batch_size, samples_min, samples_max, target_parser = sparse_tuple_from):#[(samples1, targets1, samples_len, 4)]:
            feed = {inputs: train_inputs,
                    targets: train_targets,
                    seq_len: train_seq_len}
            batch_cost, _ = session.run([cost, optimizer], feed)
            train_cost += batch_cost*batch_size
            train_ler += session.run(ler, feed_dict=feed)*batch_size

        train_cost /= num_examples
        train_ler /= num_examples

        val_feed = feed#{inputs: train_inputs,
                    #targets: train_targets,
                    #seq_len: train_seq_len}

        val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)

        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
        print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler,
                         val_cost, val_ler, time.time() - start))
    # Decoding
    d = session.run(decoded[0], feed_dict={inputs: train_inputs, seq_len : train_seq_len})
    print(d[1])
    str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
    # Replacing blank label to none
    str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
    # Replacing space label to space
    str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')

    print('Original:\n{}'.format(originals))
    print('Decoded:\n{}'.format(str_decoded))