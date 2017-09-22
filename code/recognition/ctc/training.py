import os
import signal
import time

import matplotlib
import tensorflow as tf

matplotlib.use('agg')
import numpy as np, matplotlib.pyplot as plt

from six.moves import xrange as range
from datetime import datetime

from recognition.ctc import data, testing
from utils.utils import sparse_tuple_from as sparse_tuple_from, calculate_error
from utils.utils import get_total_params_num

from . import model

ctrlc = False

def sigint_handler(signum, frame):
    print('CTRL+c pressed.\nFinishing last training epoch')
    global ctrlc
    ctrlc = True

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

def save_dataset(model_folder_name, training_data, validation_data, testing_data):
    with open('{}/{}'.format(model_folder_name, 'training.txt'), 'w+') as f:
        f.write('\n'.join([d[0] for d in training_data]))
    with open('{}/{}'.format(model_folder_name, 'testing.txt'), 'w+') as f:
        f.write('\n'.join([d[0] for d in testing_data]))
    with open('{}/{}'.format(model_folder_name, 'validation.txt'), 'w+') as f:
        f.write('\n'.join([d[0] for d in validation_data]))

# THE MAIN CODE!
def train(arguments):
    model_folder_name = '../data/umeerj/checkpoints/{}/{}'.format('_'.join([str(arg) for arg in arguments]),
                                                                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Some configs
    num_features = 13

    gpu = arguments[0]
    num_epochs = int(arguments[1])
    num_hidden = int(arguments[2])
    num_layers = int(arguments[3])
    batch_size = int(arguments[4])
    initial_learning_rate = float(arguments[5])
    momentum = float(arguments[6])
    input_dropout_keep_prob = float(arguments[7])
    output_dropout_keep_prob = float(arguments[8])
    state_dropout_keep_prob = float(arguments[9])
    affine_dropout_keep_prob = float(arguments[10])

    # Dividing the data
    training_data, training_inputs, training_targets, training_inputs_mean, training_inputs_std, \
    validation_data, validation_inputs, validation_targets, \
    testing_data, testing_inputs, testing_targets, \
    int_to_char, num_classes, num_samples = \
        data.load_data_sets('../data/umeerj/10k')
    print('{} samples without padding'.format(num_samples))

    graph = tf.Graph()
    with graph.as_default():
        inputs, targets, seq_len, input_dropout_keep, output_dropout_keep, state_dropout_keep, \
        affine_dropout_keep, cost, ler, ler2, ler3, dense_hypothesis, dense_targets, decoded, optimizer \
            = model.create_model(num_features, num_hidden, num_layers, num_classes, initial_learning_rate, momentum, batch_size)
        trainable_parameters = get_total_params_num()
        print("{} trainable parameters".format(trainable_parameters))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as session:
        signal.signal(signal.SIGINT, sigint_handler)
        os.makedirs(model_folder_name)
        #save_dataset(model_folder_name, training_data, validation_data, testing_data)

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
                    data.batch_generator(training_inputs, training_targets, batch_size, training_inputs_mean,
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
                    data.batch_generator(validation_inputs, validation_targets, batch_size, training_inputs_mean,
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
                    f.write('E: {}\n'.format(curr_epoch))
                    f.write('Tr_err: {}\nVal_err: {}\n'.format(train_ler, val_ler))
                    f.write('{} samples without padding\n{} trainable parameters\n'.format(num_samples, trainable_parameters))
                lowest_val_error = val_ler

            train_losses.append(train_cost)
            train_errors.append(train_ler)
            val_losses.append(val_cost)
            val_errors.append(val_ler)
            plot(train_losses, val_losses, train_errors, val_errors, "{}/".format(model_folder_name))

            log = "{} E: {}/{}, Tr_loss: {:.3f}, Tr_err: {:.1f}%, Val_loss: {:.3f}, Val_err: {:.1f}%, time: {:.2f} s " \
                  "- - - GPU: {}, H: {}, L: {}, BS: {}, LR: {}, M: {}, Dr-keep: {} / {} / {} / {}, "
            log = log.format(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), curr_epoch+1, num_epochs, train_cost,
                             train_ler * 100, val_cost, val_ler * 100,
                             time.time() - start, gpu, num_hidden, num_layers, batch_size, initial_learning_rate,
                             momentum, input_dropout_keep_prob, output_dropout_keep_prob,
                             state_dropout_keep_prob, affine_dropout_keep_prob)

            with open('{}/{}'.format(model_folder_name, history_fname), 'a') as f:
                f.write(log + '\n')
            print(log)

        print('Testing network.\nSaved to {}'.format(model_folder_name))
        testing.test_network(session, validation_inputs, validation_targets, batch_size, training_inputs_mean, training_inputs_std,
                             validation_data, 'validation', decoded, dense_hypothesis, inputs, seq_len, input_dropout_keep,
                             output_dropout_keep, state_dropout_keep, affine_dropout_keep, int_to_char, model_folder_name,
                             final_outcomes_fname)
        testing.test_network(session, training_inputs, training_targets, batch_size, training_inputs_mean, training_inputs_std,
                             training_data, 'training', decoded, dense_hypothesis, inputs, seq_len, input_dropout_keep,
                             output_dropout_keep, state_dropout_keep, affine_dropout_keep, int_to_char, model_folder_name,
                             final_outcomes_fname)

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
                enumerate(
                    data.batch_generator(testing_inputs, testing_targets, batch_size, training_inputs_mean,
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