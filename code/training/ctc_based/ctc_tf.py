import os, time, matplotlib, tensorflow as tf, signal
matplotlib.use('agg')
import numpy as np, matplotlib.pyplot as plt

from six.moves import xrange as range
from random import shuffle
from datetime import datetime

from data_handling.loading import ctc_targeted
from utils.utils import sparse_tuple_from as sparse_tuple_from, calculate_error
from utils.utils import get_total_params_num

ctrlc = False

def sigint_handler(signum, frame):
    print('CTRL+c pressed.\nFinishing last training epoch')
    global ctrlc
    ctrlc = True

dataset_fname = 'datasets.txt'
final_outcomes_fname = 'final_outcomes.txt'
history_fname = 'history.txt'
params_fname = 'params.txt'
error_plot_fname = 'errors.png'
loss_plot_fname = 'losses.png'

def plot(train_losses, val_losses, train_errors, val_errors, fname):
    train_loss_line, = plt.plot(np.arange(1, len(train_losses) + 1), train_losses, label='Training loss')
    val_loss_line, = plt.plot(np.arange(1, len(val_losses) + 1), val_losses, label='Validation loss')
    plt.legend(handles=[train_loss_line, val_loss_line])
    plt.xlabel('Epoch')
    plt.ylabel('CTC Loss')
    plt.grid(True)
    plt.savefig("{}{}".format(fname, loss_plot_fname))
    plt.close()

    train_err_line, = plt.plot(np.arange(1, len(train_errors) + 1), train_errors, label='Training error')
    val_err_line, = plt.plot(np.arange(1, len(val_errors) + 1), val_errors, label='Validation error')
    plt.legend(handles = [train_err_line, val_err_line])
    #plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d %'))
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid(True)
    plt.savefig("{}{}".format(fname, error_plot_fname))
    plt.close()

def divide_data(num_examples, training_part, testing_part, shuffle_count = 0, sort_by_length = False):
    SPACE_TOKEN = '<space>'

    data = ctc_targeted.load_data_from_file('../data/umeerj/data_both_mfcc.dat', [20, 13], num_examples)
    print('Loaded {} data rows'.format(len(data)))
    data = sorted(data, key=lambda x: x[2].shape[0], reverse=True)

    data = data[-num_examples:]
    print('Taking {} samples'.format(len(data)))

    for i in range(shuffle_count):
        shuffle(data)
        print('Shuffled')

    print('Totally {} samples, without padding'.format(sum([d[2].shape[0] for d in data])))

    # fs, audio = wav.read('/home/kapi/projects/research/phd/asr/data/umeerj/ume-erj/wav/JE/DOS/F01/S6_001.wav')
    # data[0] = (data[0][0], mfcc(audio, samplerate=fs), data[0][2], data[0][3])
    alphabet = ctc_targeted.get_alphabet2([d[4] for d in data])
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

    training_data = data[:int(training_part * len(data))]
    if sort_by_length:
        training_data = sorted(training_data, key= lambda x: x[2].shape[0], reverse=True)
        print('Sorting examples by descending sequence length')
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
    #validation_data = training_data
    validation_inputs = [d[2] for d in validation_data]
    validation_targets = [d[3] for d in validation_data]
    #validation_targets = np.array([np.array([10, 20, 30, 40, 50]),
    #                               np.array([0] * 1000000),
    #                                        np.array([0] * 1000000),
    #                                                 np.array([0] * 1000000),
    #                                                          np.array([0] * 1000000)])

    return training_data, training_inputs, training_targets, training_inputs_mean, training_inputs_std, validation_data, \
           validation_inputs, validation_targets, testing_data, testing_inputs, testing_targets, int_to_char, num_classes

def save_dataset(model_folder_name, training_data, validation_data, testing_data):
    with open('{}/{}'.format(model_folder_name, dataset_fname), 'w+') as f:
        f.write('\n'.join([d[0] for d in training_data]))
        f.write('\n--\n')
        f.write('\n'.join([d[0] for d in validation_data]))
        f.write('\n--\n')
        f.write('\n'.join([d[0] for d in testing_data]))

def create_model(num_features, num_hidden, num_layers, num_classes, initial_learning_rate, momentum, batch_size):
    # e.g: log filter bank or MFCC features
    # Has size [batch_size, max_stepsize, num_features], but the
    # batch_size and max_stepsize can vary along each step
    inputs = tf.placeholder(tf.float32, [None, None, num_features], name='input_samples')
    input_dropout_keep = tf.placeholder(tf.float32, name='in_dropout')
    output_dropout_keep = tf.placeholder(tf.float32, name='out_dropout')
    state_dropout_keep = tf.placeholder(tf.float32, name='state_dropout')
    affine_dropout_keep = tf.placeholder(tf.float32, name="affine_dropout")

    targets = tf.sparse_placeholder(tf.int32, name='input_targets')
    seq_len = tf.placeholder(tf.int32, [None], name='input_sequence_length')

    def lstm_cell():
        cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True, initializer=tf.contrib.layers.xavier_initializer())
        return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=output_dropout_keep,
                                             input_keep_prob=input_dropout_keep,
                                             state_keep_prob=state_dropout_keep)

    stack = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell() for _ in range(num_layers)])

    outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32, parallel_iterations=1024)

    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]
    outputs = tf.reshape(outputs, [-1, num_hidden])

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    #W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1), name='W')
    W = tf.get_variable("W", shape=[num_hidden, num_classes], initializer=tf.contrib.layers.xavier_initializer())

    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    b = tf.Variable(tf.constant(0., shape=[num_classes]))

    # Doing the affine projection
    logits = tf.matmul(outputs, W) + b
    logits = tf.nn.dropout(logits, affine_dropout_keep)

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [batch_s, -1, num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(loss, name='cost')

    optimizer = tf.train.MomentumOptimizer(initial_learning_rate,
                                           momentum).minimize(cost, name="optimization_operation")

    # Option 2: tf.nn.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    # decoded = sparse_tuple_from(np.array([[1, 2, 3, 4, 5],
    #                                [0] * 1000000,
    #                                      [0] * 1000000,
    #                                      [0] * 1000000,
    #                  [0] * 1000000]))
    # decoded = [tf.SparseTensor(decoded[0], decoded[1], decoded[2])]
    casted_hypothesis = tf.cast(decoded[0], tf.int32)

    ler = tf.reduce_mean(tf.edit_distance(casted_hypothesis, targets), name='error_rate')

    dense_targets = tf.sparse_to_dense(targets.indices, targets.dense_shape, targets.values, default_value=-1)
    dense_hypothesis = tf.sparse_to_dense(decoded[0].indices, decoded[0].dense_shape, decoded[0].values,
                                          default_value=-1)
    dense_hypothesis_lengths = [tf.to_float(tf.shape(i)[0]) for i in tf.unstack(dense_hypothesis, batch_size)]
    dense_targets_lengths = [tf.to_float(tf.shape(i)[0]) for i in tf.unstack(dense_targets, batch_size)]

    '''
    dense_hypothesis2 = []
    i = tf.constant(0)
    while_condition = lambda i: tf.less(i, tf.shape(dense_hypothesis)[0])

    def body(i):
        # do something here which you want to do in your loop
        dense_hypothesis2[i] = 5#append(tf.shape(dense_hypothesis))
        # increment i
        return [tf.add(i, 1)]

    # do the loop:
    r = tf.while_loop(while_condition, body, [i])
    #for i in tf.range(tf.to_int32(tf.shape(dense_hypothesis)[0])):
    #    dense_hypothesis2.append(tf.shape(dense_hypothesis))#tf.where(tf.equal(tf.unstack(dense_hypothesis)[i], -1)))

    tmp_indices = tf.where(tf.equal(dense_hypothesis, -1))
    #dense_hypothesis2 = tf.segment_min(tmp_indices[:, 1], tmp_indices[:, 0])
    '''

    ler2 = tf.edit_distance(casted_hypothesis, targets, False)
    ler2 = tf.reduce_sum(ler2)
    ler2 /= tf.reduce_sum(tf.maximum(dense_targets_lengths, dense_hypothesis_lengths))

    ler3 = tf.reduce_sum(tf.edit_distance(casted_hypothesis, targets, False))
    max_lengths = tf.reduce_sum(tf.maximum(dense_targets_lengths, dense_hypothesis_lengths))

    return inputs, targets, seq_len, input_dropout_keep, output_dropout_keep, state_dropout_keep, affine_dropout_keep, \
            cost, ler, ler2, ler3, dense_hypothesis, dense_targets, decoded, optimizer

def test_network(session, test_inputs, test_targets, batch_size, training_inputs_mean, training_inputs_std, data, mode,
                 decoded, dense_hypothesis, inputs, seq_len, input_dropout_keep, output_dropout_keep, state_dropout_keep,
                 affine_dropout_keep, int_to_char, model_folder_name):
    with open('{}/{}'.format(model_folder_name, final_outcomes_fname), 'a') as f:
        # Testing network
        i = 0
        f.write('---' + mode + '---\n')
        for test_inputs, _, test_seq_len, _ in \
                ctc_targeted.batch_generator(test_inputs, test_targets, batch_size, training_inputs_mean,
                                             training_inputs_std, mode=mode):
            d, dh = session.run([decoded[0], dense_hypothesis], {inputs: test_inputs,
                                         seq_len: test_seq_len,
                                         input_dropout_keep: 1,
                                         output_dropout_keep: 1,
                                         state_dropout_keep: 1,
                                         affine_dropout_keep: 1
                                         }
                            )
            dh = [d[:np.where(d == -1)[0][0] if -1 in d else len(d)] for d in dh]
            decoded_results = [''.join([int_to_char[x] for x in hypothesis]) for hypothesis in np.asarray(dh)]
            decoded_results = [s.replace('<space>', ' ') for s in decoded_results]
    #        str_decoded = ' '.join([int_to_char[x] for x in np.asarray(d[1])])
            # Replacing blank label to none
    #        str_decoded = str_decoded.replace(chr(ord('z') + 1), '<BLANK>')
            # Replacing space label to space
    #        str_decoded = str_decoded.replace('<space>', ' ')
    #        print('Decoded:\n{}'.format(str_decoded))

            for h in decoded_results:
                f.write('{} -> {}\n'.format(data[i][4], h))
                i += 1

# THE MAIN CODE!
def train(arguments):
    model_folder_name = '../data/umeerj/checkpoints/{}/{}'.format('_'.join([str(arg) for arg in arguments]), str(datetime.now()))

    # Some configs
    num_features = 13

    gpu = arguments[0]
    num_epochs = int(arguments[1])
    num_hidden = int(arguments[2])
    num_layers = int(arguments[3])
    batch_size = int(arguments[4])
    initial_learning_rate = float(arguments[5])
    momentum = float(arguments[6])
    num_examples = int(arguments[7])
    input_dropout_keep_prob = float(arguments[8])
    output_dropout_keep_prob = float(arguments[9])
    state_dropout_keep_prob = float(arguments[10])
    affine_dropout_keep_prob = float(arguments[11])
    training_part = float(arguments[12])
    testing_part = float(arguments[13])
    shuffle_count = int(arguments[14])
    sort_by_length = bool(int(arguments[15]))

    # Dividing the data
    training_data, training_inputs, training_targets, training_inputs_mean, training_inputs_std, \
    validation_data, validation_inputs, validation_targets, \
    testing_data, testing_inputs, testing_targets, \
    int_to_char, num_classes = divide_data(num_examples, training_part, testing_part, shuffle_count, sort_by_length)

    save_dataset(model_folder_name, training_data, validation_data, testing_data)

    graph = tf.Graph()
    with graph.as_default():
        inputs, targets, seq_len, input_dropout_keep, output_dropout_keep, state_dropout_keep, \
        affine_dropout_keep, cost, ler, ler2, ler3, dense_hypothesis, dense_targets, decoded, optimizer \
            = create_model(num_features, num_hidden, num_layers, num_classes, initial_learning_rate, momentum, batch_size)
        print("Totally {} trainable parameters".format(get_total_params_num()))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as session:
        signal.signal(signal.SIGINT, sigint_handler)
        os.makedirs(model_folder_name)

        # Initializate the weights and biases
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        lowest_val_error = 2
        val_errors = list()
        train_errors = list()
        val_losses = list()
        train_losses = list()

        saver = tf.train.Saver()
        for curr_epoch in range(num_epochs):
            train_cost = train_ler = train_max_lengths = 0
            start = time.time()

            for train_inputs, train_targets, train_seq_len, _ in \
                    ctc_targeted.batch_generator(training_inputs, training_targets, batch_size, training_inputs_mean,
                                                 training_inputs_std, target_parser = sparse_tuple_from):
                if ctrlc: break

                feed = {inputs: train_inputs,
                        targets: train_targets,
                        seq_len: train_seq_len,
                        input_dropout_keep: input_dropout_keep_prob,
                        output_dropout_keep: output_dropout_keep_prob,
                        state_dropout_keep: state_dropout_keep_prob,
                        affine_dropout_keep: affine_dropout_keep_prob
                        }

                session.run(optimizer, feed)

                batch_cost, dh, dt = session.run([cost, dense_hypothesis, dense_targets], feed)
                train_cost += batch_cost

                levsum, lensum = calculate_error(dh, dt)
                train_ler += levsum
                train_max_lengths += lensum
            if ctrlc: break

            train_ler /= train_max_lengths
            train_cost /= len(training_inputs) // batch_size

            val_cost = val_ler = val_max_lengths = 0
            for v_inputs, v_targets, v_seq_len, _ in \
                    ctc_targeted.batch_generator(validation_inputs, validation_targets, batch_size, training_inputs_mean,
                                                 training_inputs_std, target_parser = sparse_tuple_from, mode='validation'):
                if ctrlc: break

                v_cost, dh, dt = session.run([cost, dense_hypothesis, dense_targets],
                                             {inputs: v_inputs,
                                              targets: v_targets,
                                              seq_len: v_seq_len,
                                              input_dropout_keep: 1,
                                              output_dropout_keep: 1,
                                              state_dropout_keep: 1,
                                              affine_dropout_keep: 1
                                              }
                                             )

                val_cost += v_cost

                levsum, lensum = calculate_error(dh, dt)
                val_max_lengths += lensum
                val_ler += levsum
            if ctrlc: break

            val_ler /= val_max_lengths
            val_cost /= len(validation_inputs) // batch_size

            if val_ler < lowest_val_error:
                saver.save(session, model_folder_name + '/model')
                with open('{}/{}'.format(model_folder_name, params_fname), 'w+') as f:
                    f.write(str(curr_epoch) + '\n')
                    f.write(str(val_ler) + '\n' + str(train_ler) + '\n')
                lowest_val_error = val_ler

            train_losses.append(train_cost)
            train_errors.append(train_ler)
            val_losses.append(val_cost)
            val_errors.append(val_ler)
            plot(train_losses, val_losses, train_errors, val_errors, "{}/".format(model_folder_name))

            log = "E: {}/{}, Tr_loss: {:.3f}, Tr_err: {:.1f}%, Val_loss: {:.3f}, Val_err: {:.1f}%, time: {:.2f} s " \
                  "- - - GPU: {}, H: {}, L: {}, BS: {}, LR: {}, M: {}, Ex: {}, Dr-keep: {} / {} / {} / {}, " \
                  "Data: {:.2f} / {:.2f} / {:.2f}, Shuffle: {}, Sort by length: {}"
            log = log.format(curr_epoch+1, num_epochs, train_cost, train_ler * 100, val_cost, val_ler * 100,
                             time.time() - start, gpu, num_hidden, num_layers, batch_size, initial_learning_rate,
                             momentum, num_examples, input_dropout_keep_prob, output_dropout_keep_prob,
                             state_dropout_keep_prob, affine_dropout_keep_prob, training_part,
                             1 - training_part - testing_part, testing_part, shuffle_count, sort_by_length)

            with open('{}/{}'.format(model_folder_name, history_fname), 'a') as f:
                f.write(log + '\n')
            print(log)

        print('Testing network.\nSaved to {}'.format(model_folder_name))
        test_network(session, validation_inputs, validation_targets, batch_size, training_inputs_mean, training_inputs_std,
                     validation_data, 'validation', decoded, dense_hypothesis, inputs, seq_len, input_dropout_keep,
                     output_dropout_keep, state_dropout_keep, affine_dropout_keep, int_to_char, model_folder_name)
        test_network(session, testing_inputs, testing_targets, batch_size, training_inputs_mean, training_inputs_std,
                     testing_data, 'testing', decoded, dense_hypothesis, inputs, seq_len, input_dropout_keep,
                     output_dropout_keep, state_dropout_keep, affine_dropout_keep, int_to_char, model_folder_name)
        test_network(session, training_inputs, training_targets, batch_size, training_inputs_mean, training_inputs_std,
                     training_data, 'training', decoded, dense_hypothesis, inputs, seq_len, input_dropout_keep,
                     output_dropout_keep, state_dropout_keep, affine_dropout_keep, int_to_char, model_folder_name)

def load_and_test():
    training_inputs, training_targets, training_inputs_mean, training_inputs_std, \
    validation_data, validation_inputs, validation_targets, \
    testing_data, testing_inputs, testing_targets, \
    int_to_char, num_classes = divide_data(10000, 0.7, 0.15)
    batch_size = 25

    saver = tf.train.import_meta_graph('../data/umeerj/checkpoints/2000_1000_3_25_0.0005_0.9_10000_1_0.7_0.15/model.meta')
    with tf.Session() as sess:
        # First let's load meta graph and restore weights
        #saver = tf.train.import_meta_graph('my_test_model-0')
        saver.restore(sess, tf.train.latest_checkpoint('../data/umeerj/checkpoints/2000_1000_3_25_0.0005_0.9_10000_1_0.7_0.15/'))
        graph = tf.get_default_graph()
        decoded = graph.get_tensor_by_name('CTCGreedyDecoder:1')
        input_samples = graph.get_tensor_by_name('input_samples:0')
        dropout_keep = graph.get_tensor_by_name('dropout:0')
        input_sequence_length = graph.get_tensor_by_name('input_sequence_length:0')
        #print('\n'.join([n.name for n in graph.as_graph_def().node]))

        for i, (test_input, _, test_seq_len, _) in \
                enumerate(ctc_targeted.batch_generator(testing_inputs, testing_targets, batch_size, training_inputs_mean,
                                                       training_inputs_std)):
            d = sess.run(decoded, feed_dict={
                input_samples: test_input, input_sequence_length: test_seq_len, dropout_keep:1
            })
            #str_decoded = ''.join([chr(x) for x in np.asarray(d) + FIRST_INDEX])
            str_decoded = ''.join([int_to_char[x] for x in np.asarray(d)])
            # Replacing blank label to none
            str_decoded = str_decoded.replace(chr(ord('z') + 1), '<BLANK>')
            # Replacing space label to space
            str_decoded = str_decoded.replace('<space>', ' ')
            print('Decoded:\n{}'.format(str_decoded))