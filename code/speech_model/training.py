##Producing Training/Testing inputs+output
from random import random
import os, tensorflow as tf, numpy as np

# Random initial angles
angle1 = random()
angle2 = random()

# The total 2*pi cycle would be divided into 'frequency'
# number of steps
frequency1 = 300
frequency2 = 200
# This defines how many steps ahead we are trying to predict
lag = 23

def get_sample():
    """
    Returns a [[sin value, cos value]] input.
    """
    global angle1, angle2
    angle1 += 2 * np.pi / float(frequency1)
    angle2 += 2 * np.pi / float(frequency2)
    angle1 %= 2 * np.pi
    angle2 %= 2 * np.pi
    return np.array([np.array([
        5 + 5 * np.sin(angle1) + 10 * np.cos(angle2),
        7 + 7 * np.sin(angle2) + 14 * np.cos(angle1)])])


sliding_window = []

for i in range(lag - 1):
    sliding_window.append(get_sample())


def get_pair():
    """
    Returns an (current, later) pair, where 'later' is 'lag'
    steps ahead of the 'current' on the wave(s) as defined by the
    frequency.
    """

    global sliding_window
    sliding_window.append(get_sample())
    input_value = sliding_window[0]
    output_value = sliding_window[-1]
    sliding_window = sliding_window[1:]
    return input_value, output_value


# Input Params
input_dim = 2

# To maintain state
last_value = np.array([0 for i in range(input_dim)])
last_derivative = np.array([0 for i in range(input_dim)])


def get_total_input_output():
    """
    Returns the overall Input and Output as required by the model.
    The input is a concatenation of the wave values, their first and
    second derivatives.
    """
    global last_value, last_derivative
    raw_i, raw_o = get_pair()
    raw_i = raw_i[0]
    l1 = list(raw_i)
    derivative = raw_i - last_value
    l2 = list(derivative)
    last_value = raw_i
    l3 = list(derivative - last_derivative)
    last_derivative = derivative
    return np.array([l1 + l2 + l3]), raw_o


# Input Params
input_dim = 2

##The Input Layer as a Placeholder
# Since we will provide data sequentially, the 'batch size'
# is 1.
input_layer = tf.placeholder(tf.float32, [1, input_dim * 3])


##The LSTM Layer-1
# The LSTM Cell initialization
lstm_layer1 = tf.contrib.rnn.BasicLSTMCell(input_dim * 3)
# The LSTM state as a Variable initialized to zeroes
#lstm_state1 = tf.Variable(tf.zeros([1, lstm_layer1.state_size]))
lstm_state1 = (tf.Variable(tf.zeros([1, input_dim * 3])),) * 2

#os._exit(0)
# Connect the input layer and initial LSTM state to the LSTM cell
lstm_output1, lstm_state_output1 = lstm_layer1(input_layer, lstm_state1,
                                               scope='LSTM1')
# The LSTM state will get updated
#lstm_update_op1 = lstm_state1.assign(lstm_state_output1)
lstm_update_op1 = tf.assign(lstm_state1[0], lstm_state_output1[0])
lstm_update_op2 = tf.assign(lstm_state1[1], lstm_state_output1[1])

##The Regression-Output Layer1
# The Weights and Biases matrices first
output_W1 = tf.Variable(tf.truncated_normal([input_dim * 3, input_dim]))
output_b1 = tf.Variable(tf.zeros([input_dim]))
# Compute the output
final_output = tf.matmul(lstm_output1, output_W1) + output_b1

##Input for correct output (for training)
correct_output = tf.placeholder(tf.float32, [1, input_dim])

##Calculate the Sum-of-Squares Error
error = tf.pow(final_output - correct_output, 2)

##The Optimizer
# Adam works best
train_step = tf.train.AdamOptimizer(0.0006).minimize(error)

##Session

with tf.Session() as sess:
    # Initialize all Variables
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()

    ##Training

    actual_output1 = []
    actual_output2 = []
    network_output1 = []
    network_output2 = []
    x_axis = []

    for i in range(0, 80000, 1):
        input_v, output_v = get_total_input_output()
        #'''
        _, _, _, network_output, err = sess.run([lstm_update_op1, lstm_update_op2,
                                         train_step,
                                         final_output, error],
                                        feed_dict={
                                            input_layer: input_v,
                                            correct_output: output_v})

        #'''
        #network_output = [[0, 0]]
        if i % 2000 != 0:
            continue
        network_output1.append(network_output[0][0])
        network_output2.append(network_output[0][1])
        actual_output1.append(output_v[0][0])
        actual_output2.append(output_v[0][1])
        x_axis.append(i)
        print('{} / {}, {}'.format(i, 80000, err))

import matplotlib.pyplot as plt

plt.plot(x_axis, network_output1, 'r-', x_axis, actual_output1, 'b-')
plt.show()
plt.plot(x_axis, network_output2, 'r-', x_axis, actual_output2, 'b-')
plt.show()

os._exit(0)









import os
import signal
import time

import matplotlib
import tensorflow as tf

matplotlib.use('agg')
import numpy as np, matplotlib.pyplot as plt

from six.moves import xrange as range
from datetime import datetime

from recognition.ctc import testing, data
from . import data as ldata
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
    batch_size = 1#int(arguments[2])
    initial_learning_rate = float(arguments[3])
    num_epochs = 100000#int(arguments[4])
    input_dropout_keep_prob = float(arguments[5])
    output_dropout_keep_prob = float(arguments[6])
    state_dropout_keep_prob = float(arguments[7])
    affine_dropout_keep_prob = float(arguments[8])
    dataset = arguments[9]

    # Dividing the data
    '''
    training_data, training_inputs, training_targets, training_inputs_mean, training_inputs_std, \
    validation_data, validation_inputs, validation_targets, \
    testing_data, testing_inputs, testing_targets, \
    int_to_char, num_classes, num_samples = \
        data.load_data_sets('../data/umeerj/{}'.format(dataset))
    print('{} samples without padding'.format(num_samples))
    '''

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        if training_mode == 'new':
            num_hidden = int(arguments[10])
            num_layers = int(arguments[11])
            momentum = float(arguments[12])
            model_folder_name = '../data/umeerj/checkpoints/speech_model/{}/{}'.format('_'.join([str(arg) for arg in arguments]),
                                                                          datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            inputs, targets, seq_len, input_dropout_keep, output_dropout_keep, state_dropout_keep, \
            cost, optimizer, learning_rate, pred \
               = model.create_model(num_features, num_hidden, num_layers, momentum, batch_size)
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
        else:
            model_folder_name = old_foolder_name = arguments[10]
            inputs, targets, seq_len, input_dropout_keep, output_dropout_keep, state_dropout_keep, \
            affine_dropout_keep, cost, ler, ler2, ler3, dense_hypothesis, dense_targets, decoded, optimizer \
                = model.load_existing_model(model_folder_name, session)
            model_folder_name += '/' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        os.makedirs(model_folder_name)
        trainable_parameters = get_total_params_num()
        print("{} trainable parameters".format(trainable_parameters))

        signal.signal(signal.SIGINT, sigint_handler)

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
                    ldata.single_batch_generator(batch_size):
                    #data.batch_generator(training_inputs, training_targets, batch_size, training_inputs_mean,
                    #                     training_inputs_std):#, target_parser = sparse_tuple_from):
                if ctrlc: break

                feed = {inputs: train_inputs,
                        targets: train_targets,
                        seq_len: train_seq_len,
                        input_dropout_keep: input_dropout_keep_prob,
                        output_dropout_keep: output_dropout_keep_prob,
                        state_dropout_keep: state_dropout_keep_prob,
#                        affine_dropout_keep: affine_dropout_keep_prob,
                        }
                if 'learning_rate' in locals():
                    feed[learning_rate] = initial_learning_rate

                session.run(optimizer, feed)
#                batch_cost, dh, dt = session.run([cost, dense_hypothesis, dense_targets], feed)
#                train_cost += batch_cost

#                levsum, lensum = calculate_error(dh, dt)
#                train_ler += levsum
#                train_max_lengths += lensum
            if ctrlc: break

 #           train_ler /= train_max_lengths
 #           train_cost /= len(training_inputs) // batch_size

            val_cost = val_ler = val_max_lengths = 0
            for v_inputs, v_targets, v_seq_len, _ in \
                    ldata.single_batch_generator(batch_size):
                    #data.batch_generator(training_inputs, training_targets, batch_size, training_inputs_mean,
                    #                     training_inputs_std):#, target_parser = sparse_tuple_from):
                if ctrlc: break

                v_cost, preds = session.run([cost, pred],
                                             {
                                                 inputs: v_inputs,
                                                 targets: v_targets,
                                                 seq_len: v_seq_len,
                                                 input_dropout_keep: 1,
                                                 output_dropout_keep: 1,
                                                 state_dropout_keep: 1,
 #                                                affine_dropout_keep: 1
                                             }
                                            )
                print(v_inputs)
                print(v_targets)
                print(preds)

            val_cost += v_cost

            if ctrlc: break

            #val_ler /= val_max_lengths
            #val_cost /= len(validation_inputs) // batch_size

            if val_ler < lowest_val_error:
                saver.save(session, model_folder_name + '/model')
                with open('{}/{}'.format(model_folder_name, params_fname), 'w+') as f:
                    f.write('E: {}\n'.format(curr_epoch))
                    f.write('Tr_err: {}\nVal_err: {}\n'.format(train_ler, val_ler))
#yyhh                    f.write('{} samples without padding\n{} trainable parameters\n'.format(num_samples, trainable_parameters))
                lowest_val_error = val_ler

            train_losses.append(train_cost)
            train_errors.append(train_ler)
            val_losses.append(val_cost)
            val_errors.append(val_ler)
            plot(train_losses, val_losses, train_errors, val_errors, "{}/".format(model_folder_name))

            if training_mode == 'new':
                log = "{} E: {}/{}, Tr_loss: {:.3f}, Tr_err: {:.1f}%, Val_loss: {:.3f}, Val_err: {:.1f}%, time: {:.2f} s " \
                      "- - - GPU: {}, H: {}, L: {}, BS: {}, LR: {}, M: {}, Dr-keep: {} / {} / {} / {}, "
                log = log.format(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), curr_epoch+1, num_epochs, train_cost,
                                 train_ler * 100, val_cost, val_ler * 100,
                                 time.time() - start, gpu, num_hidden, num_layers, batch_size, initial_learning_rate,
                                 momentum, input_dropout_keep_prob, output_dropout_keep_prob,
                                 state_dropout_keep_prob, affine_dropout_keep_prob)
            else:
                log = "{} E: {}/{}, Tr_loss: {:.3f}, Tr_err: {:.1f}%, Val_loss: {:.3f}, Val_err: {:.1f}%, time: {:.2f} s " \
                      "- - - GPU: {}, BS: {}, LR: {}, Dr-keep: {} / {} / {} / {}, "
                log = log.format(datetime.now().strftime("%Y/%m/%d %H:%M:%S"), curr_epoch+1, num_epochs, train_cost,
                                 train_ler * 100, val_cost, val_ler * 100,
                                 time.time() - start, gpu, batch_size, initial_learning_rate,
                                 input_dropout_keep_prob, output_dropout_keep_prob,
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