import time, os
import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np

from six.moves import xrange as range
from python_speech_features import mfcc
from random import shuffle

from feature_extraction import extraction
from data_handler.loaders import tf_loader
from .utils import maybe_download as maybe_download
from .utils import sparse_tuple_from as sparse_tuple_from
from .utils import get_total_params_num

# THE MAIN CODE!
def train(gpu, arguments):
    model_folder_name = '../data/umeerj/checkpoints/{}'.format('_'.join([str(arg) for arg in arguments]))
    if not os.path.isdir(model_folder_name):
        os.makedirs(model_folder_name)

    # Constants
    SPACE_TOKEN = '<space>'
    # SPACE_INDEX = 0
    # FIRST_INDEX = ord('a') - 1  # 0 is reserved to space

    # Some configs
    num_features = 13
    # Accounting the 0th indice +  space + blank label = 28 characters
    num_classes = ord('z') - ord('a') + 1 + 1 + 1

    # Hyper-parameters
    num_epochs = int(arguments[0])
    num_hidden = int(arguments[1])
    num_layers = int(arguments[2])
    batch_size = int(arguments[3])
    initial_learning_rate = float(arguments[4])
    momentum = float(arguments[5])
    num_examples = int(arguments[6])
    dropout_keep_prob = float(arguments[7])

    # Loading the data

    data = tf_loader.load_data_from_file('../data/umeerj/data_both_mfcc.dat', [20, 13], num_examples)
    print('Loaded {} data rows'.format(len(data)))
    for i in range(3):
        shuffle(data)
        print('Shuffled')
    data = data[:num_examples]
    print('Taking {} samples'.format(len(data)))

    # fs, audio = wav.read('/home/kapi/projects/research/phd/asr/data/umeerj/ume-erj/wav/JE/DOS/F01/S6_001.wav')
    # data[0] = (data[0][0], mfcc(audio, samplerate=fs), data[0][2], data[0][3])
    alphabet = tf_loader.get_alphabet2([d[4] for d in data])
    int_to_char = {i: char for i, char in enumerate([SPACE_TOKEN] + alphabet)}
    char_to_int = {char: i for i, char in int_to_char.items()}
    num_classes = len(int_to_char) + 1

    all_targets = [d[3] for d in data]
    all_targets = [' '.join(t.strip().lower().split(' '))
                     #.replace('.', '').replace('-', '').replace("'", '').replace(':', '').replace(',', '').
                     #   replace('?', '').replace('!', '')
                 for t in all_targets]
    all_targets = [target.replace(' ', '  ') for target in all_targets]
    all_targets = [target.split(' ') for target in all_targets]
    all_targets = [np.hstack([SPACE_TOKEN if x == '' else list(x) for x in target]) for target in all_targets]
    # all_targets = [np.asarray([SPACE_INDEX if x == SPACE_TOKEN else ord(x) - FIRST_INDEX for x in target])
    #               for target in all_targets]
    all_targets = [np.asarray([char_to_int[x] for x in target]) for target in all_targets]
    data = [(d[0], d[1], d[2], all_targets[i], d[4]) for i, d in enumerate(data)]

    training_part = 0.8
    testing_part = 0.1

    training_data = data[:int(training_part * len(data))]
    training_inputs = [d[2] for d in training_data]#all_inputs[:int(training_part * len(all_inputs))]
    training_targets = [d[3] for d in training_data]#all_targets[:int(training_part * len(all_inputs))]
    training_inputs_mean = np.sum([np.sum(s) for s in training_inputs]) / np.sum([np.size(s) for s in training_inputs])
    training_inputs_std = np.sqrt(np.sum([np.sum(np.power(s - training_inputs_mean, 2)) for s in training_inputs]) /
                             np.sum([np.size(s) for s in training_inputs]))

    testing_data = data[int(training_part * len(data)):int((training_part + testing_part) * len(data))]
    #testing_data = training_data
    testing_inputs = [d[2] for d in testing_data]#all_inputs[int(training_part * len(all_inputs)):]
    testing_targets = [d[3] for d in testing_data]#all_targets[int(training_part * len(all_inputs)):]

    validation_data = data[int((training_part + testing_part) * len(data)):]
    validation_inputs = [d[2] for d in validation_data]
    validation_targets = [d[3] for d in validation_data]

    graph = tf.Graph()
    with graph.as_default():
        # e.g: log filter bank or MFCC features
        # Has size [batch_size, max_stepsize, num_features], but the
        # batch_size and max_stepsize can vary along each step
        inputs = tf.placeholder(tf.float32, [None, None, num_features], name='input_samples')
        dropout_keep = tf.placeholder(tf.float32, name='dropout')
        # Here we use sparse_placeholder that will generate a
        # SparseTensor required by ctc_loss op.
        targets = tf.sparse_placeholder(tf.int32, name='input_targets')

        # 1d array of size [batch_size]
        seq_len = tf.placeholder(tf.int32, [None], name='input_sequence_length')

        # Defining the cell
        # Can be:
        #   tf.nn.rnn_cell.RNNCell
        #   tf.nn.rnn_cell.GRUCell
        def lstm_cell():
            cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout_keep)

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
                                            stddev=0.1), name='W')
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
        cost = tf.reduce_mean(loss, name = 'cost')

        optimizer = tf.train.MomentumOptimizer(initial_learning_rate,
                                               momentum).minimize(cost, name="optimization_operation")

        # Option 2: tf.nn.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

        # Inaccuracy: label error rate
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                              targets), name='error_rate')

        print("{} trainable parameters".format(get_total_params_num()))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as session:
        # Initializate the weights and biases
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        saved = True

        #test_batch_generator = tf_loader.batch_generator(test_inputs, test_targets, batch_size, training_inputs_mean,
                    # training_inputs_std, target_parser=sparse_tuple_from, mode='testing')
        lowest_val_error = 1.1
        for curr_epoch in range(num_epochs):
            train_cost = train_ler =  0
            start = time.time()

            for train_inputs, train_targets, train_seq_len, _ in \
                    tf_loader.batch_generator(training_inputs, training_targets, batch_size, training_inputs_mean,
                                              training_inputs_std, target_parser = sparse_tuple_from):
                feed = {inputs: train_inputs,
                        targets: train_targets,
                        seq_len: train_seq_len,
                        dropout_keep: dropout_keep_prob}
                batch_cost, _, lerr = session.run([cost, optimizer, ler], feed)
                train_cost += batch_size * batch_cost
                #lerr = session.run(ler, feed_dict=feed)
                train_ler += lerr*batch_size

            train_cost /= len(training_data)
            train_ler /= len(training_data)

            val_cost = val_ler = 0
            for v_inputs, v_targets, v_seq_len, _ in \
                    tf_loader.batch_generator(validation_inputs, validation_targets, batch_size, training_inputs_mean,
                                              training_inputs_std, target_parser=sparse_tuple_from, mode='validation'):
                val_feed = {inputs: v_inputs,
                            targets: v_targets,
                            seq_len: v_seq_len,
                            dropout_keep:1}
                v_cost, v_ler = session.run([cost, ler], feed_dict=val_feed)
                val_cost += v_cost*batch_size
                val_ler += v_ler*batch_size
            val_cost /= len(validation_data)
            val_ler /= len(validation_data)

            log = "E: {}/{}, Tr_cost: {:.3f}, Tr_err: {:.3f}, Val_cost: {:.3f}, " \
                  "Val_err: {:.3f}, time: {:.3f} s - - - GPU: {}, H: {}, L: {}, BS: {}, LR: {}, M: {}, Ex: {}, Dr-keep: {}"
            print(log.format(curr_epoch+1, num_epochs, train_cost, train_ler,
                             val_cost, val_ler, time.time() - start, gpu, num_hidden, num_layers, batch_size,
                             initial_learning_rate, momentum, num_examples, dropout_keep_prob))

            if val_ler < lowest_val_error:
                saver.save(session, model_folder_name + '/model')
                with open(model_folder_name + '/params.txt', 'w+') as f:
                    f.write(str(val_ler) + '\n' + str(train_ler) + '\n')
                    f.write('\n'.join([d[0] for d in validation_data]))
                    f.write('\n--\n')
                    f.write('\n'.join([d[0] for d in testing_data]))
                lowest_val_error = val_ler

        # Testing network
        print('Validation:\n{}'.format([d[4] for d in validation_data]))#originals[int(training_part * len(all_inputs)):]))
        for i, (test_inputs, _, test_seq_len, _) in \
                enumerate(tf_loader.batch_generator(validation_inputs, validation_targets, batch_size, training_inputs_mean,
                                                    training_inputs_std, mode='validation')):
            d = session.run(decoded[0], feed_dict={
                inputs: test_inputs,
                seq_len : test_seq_len,
                dropout_keep: 1
            })
            #str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
            str_decoded = ''.join([int_to_char[x] for x in np.asarray(d[1])])
            # Replacing blank label to none
            str_decoded = str_decoded.replace(chr(ord('z') + 1), '<BLANK>')
            # Replacing space label to space
            str_decoded = str_decoded.replace('<space>', ' ')
            print('Decoded:\n{}'.format(str_decoded))

        # Testing network
        print('Test:\n{}'.format([d[4] for d in testing_data]))#originals[int(training_part * len(all_inputs)):]))
        for i, (test_inputs, _, test_seq_len, _) in \
                enumerate(tf_loader.batch_generator(testing_inputs, testing_targets, batch_size, training_inputs_mean,
                                                    training_inputs_std, mode='testing')):
            d = session.run(decoded[0], feed_dict={
                inputs: test_inputs, seq_len : test_seq_len,
                dropout_keep: 1
            })
            #str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
            str_decoded = ''.join([int_to_char[x] for x in np.asarray(d[1])])
            # Replacing blank label to none
            str_decoded = str_decoded.replace(chr(ord('z') + 1), '<BLANK>')
            # Replacing space label to space
            str_decoded = str_decoded.replace('<space>', ' ')
            print('Decoded:\n{}'.format(str_decoded))

def load_and_test():
    saver = tf.train.import_meta_graph('my_test_model.meta')
    with tf.Session() as sess:
        # First let's load meta graph and restore weights
        #saver = tf.train.import_meta_graph('my_test_model-0')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        graph = tf.get_default_graph()
        decoded = graph.get_tensor_by_name('CTCGreedyDecoder:1')
        input_samples = graph.get_tensor_by_name('input_samples:0')
        input_sequence_length = graph.get_tensor_by_name('input_sequence_length:0')
        #print('\n'.join([n.name for n in graph.as_graph_def().node]))

        for i, (test_input, _, test_seq_len, _) in \
                enumerate(tf_loader.batch_generator(test_inputs, test_targets, batch_size, all_inputs_mean,
                                                    all_inputs_std)):
            d = sess.run(decoded, feed_dict={
                input_samples: test_input, input_sequence_length: test_seq_len
            })
            #str_decoded = ''.join([chr(x) for x in np.asarray(d) + FIRST_INDEX])
            str_decoded = ''.join([int_to_char[x] for x in np.asarray(d)])
            # Replacing blank label to none
            str_decoded = str_decoded.replace(chr(ord('z') + 1), '<BLANK>')
            # Replacing space label to space
            str_decoded = str_decoded.replace('<space>', ' ')
            print('Decoded:\n{}'.format(str_decoded))

