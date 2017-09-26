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

final_outcomes_validation_fname = 'validation_outcomes.txt'
final_outcomes_training_fname = 'training_outcomes.txt'
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

# THE MAIN CODE!
def train(arguments):
    # Some configs
    num_features = 13

    gpu = arguments[0]
    training_mode = arguments[1]
    batch_size = int(arguments[2])
    initial_learning_rate = float(arguments[3])
    num_epochs = int(arguments[4])
    input_dropout_keep_prob = float(arguments[5])
    output_dropout_keep_prob = float(arguments[6])
    state_dropout_keep_prob = float(arguments[7])
    affine_dropout_keep_prob = float(arguments[8])
    if training_mode == 'new':
        num_hidden = int(arguments[9])
        num_layers = int(arguments[10])
        momentum = float(arguments[11])

    # Dividing the data
    training_data, training_inputs, training_targets, training_inputs_mean, training_inputs_std, \
    validation_data, validation_inputs, validation_targets, \
    testing_data, testing_inputs, testing_targets, \
    int_to_char, num_classes, num_samples = \
        data.load_data_sets('../data/umeerj/10k')
    print('{} samples without padding'.format(num_samples))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        if training_mode == 'new':
            model_folder_name = '../data/umeerj/checkpoints/{}/{}'.format('_'.join([str(arg) for arg in arguments]),
                                                                          datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            inputs, targets, seq_len, input_dropout_keep, output_dropout_keep, state_dropout_keep, \
            affine_dropout_keep, cost, ler, ler2, ler3, dense_hypothesis, dense_targets, decoded, optimizer \
                = model.create_model(num_features, num_hidden, num_layers, num_classes, initial_learning_rate, momentum,
                                     batch_size)
            os.makedirs(model_folder_name)
        else:
            model_folder_name = arguments[9]
            inputs, targets, seq_len, input_dropout_keep, output_dropout_keep, state_dropout_keep, \
            affine_dropout_keep, cost, ler, ler2, ler3, dense_hypothesis, dense_targets, decoded, optimizer \
                = model.load_existing_model(model_folder_name, session)
            model_folder_name += '/' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trainable_parameters = get_total_params_num()
        print("{} trainable parameters".format(trainable_parameters))

        signal.signal(signal.SIGINT, sigint_handler)

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
                                             {
                                                 inputs: v_inputs,
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
                             final_outcomes_validation_fname)
        testing.test_network(session, training_inputs, training_targets, batch_size, training_inputs_mean, training_inputs_std,
                             training_data, 'training', decoded, dense_hypothesis, inputs, seq_len, input_dropout_keep,
                             output_dropout_keep, state_dropout_keep, affine_dropout_keep, int_to_char, model_folder_name,
                             final_outcomes_training_fname)