##Producing Training/Testing inputs+output
from numpy import array, sin, cos, pi
from random import random

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
    angle1 += 2 * pi / float(frequency1)
    angle2 += 2 * pi / float(frequency2)
    angle1 %= 2 * pi
    angle2 %= 2 * pi
    return array([array([
        5 + 5 * sin(angle1) + 10 * cos(angle2),
        7 + 7 * sin(angle2) + 14 * cos(angle1)])])


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
last_value = array([0 for i in range(input_dim)])
last_derivative = array([0 for i in range(input_dim)])


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
    return array([l1 + l2 + l3]), raw_o


# Imports
import tensorflow as tf

# Input Params
input_dim = 2

##The Input Layer as a Placeholder
# Since we will provide data sequentially, the 'batch size'
# is 1.
input_layer = tf.placeholder(tf.float32, [1, input_dim * 3])

##The LSTM Layer-1
#The LSTM Cell initialization
lstm_layer1 = tf.contrib.rnn.BasicLSTMCell(input_dim*3)
#The LSTM state as a Variable initialized to zeroes
lstm_state1 = (tf.Variable(tf.zeros([1, input_dim * 3])),) * 2
#Connect the input layer and initial LSTM state to the LSTM cell
lstm_output1, lstm_state_output1 = lstm_layer1(input_layer, lstm_state1,
                                              scope="LSTM1")
#The LSTM state will get updated
lstm_update_op1 = tf.assign(lstm_state1[0], lstm_state_output1[0])
lstm_update_op2 = tf.assign(lstm_state1[1], lstm_state_output1[1])

##The Regression-Output Layer1
#The Weights and Biases matrices first
output_W1 = tf.Variable(tf.truncated_normal([input_dim*3, input_dim]))
output_b1 = tf.Variable(tf.zeros([input_dim]))
#Compute the output
final_output = tf.matmul(lstm_output1, output_W1) + output_b1

##Input for correct output (for training)
correct_output = tf.placeholder(tf.float32, [1, input_dim])

##Calculate the Sum-of-Squares Error
error = tf.reduce_sum(tf.pow(final_output - correct_output, 2))

##The Optimizer
#Adam works best
train_step = tf.train.AdamOptimizer(0.0006).minimize(error)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    ##Training

    actual_output1 = []
    actual_output2 = []
    network_output1 = []
    network_output2 = []
    x_axis = []

    for i in range(80000):
        input_v, output_v = get_total_input_output()
        _, network_output, err = sess.run([#lstm_update_op1, lstm_update_op2,
                                         train_step,
                                         final_output, error],
                                        feed_dict={
                                            input_layer: input_v,
                                            correct_output: output_v})

        if i % 2000 == 0:
            print('{} / {}. Error: {}'.format(i, 80000, err))
            actual_output1.append(output_v[0][0])
            actual_output2.append(output_v[0][1])
            network_output1.append(network_output[0][0])
            network_output2.append(network_output[0][1])
            x_axis.append(i)

    import matplotlib.pyplot as plt

    plt.plot(x_axis, network_output1, 'r-', x_axis, actual_output1, 'b-')
    plt.show()
    plt.plot(x_axis, network_output2, 'r-', x_axis, actual_output2, 'b-')
    plt.show()