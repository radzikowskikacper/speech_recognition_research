import tensorflow as tf


def create_model(num_features, num_hidden, num_layers, momentum, batch_size):
    ##The Input Layer as a Placeholder
    # Since we will provide data sequentially, the 'batch size'
    # is 1.
    inputs = tf.placeholder(tf.float32, [batch_size, None, num_features])
    seq_len = tf.placeholder(tf.int32, [batch_size], name='input_sequence_length')
    input_dropout_keep = tf.placeholder(tf.float32, name='in_dropout')
    output_dropout_keep = tf.placeholder(tf.float32, name='out_dropout')
    state_dropout_keep = tf.placeholder(tf.float32, name='state_dropout')
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    input_layer = tf.placeholder(tf.float32, [1, num_features])

    ##The LSTM Layer-1
    # The LSTM Cell initialization
    #lstm_layer1 = tf.contrib.rnn.BasicLSTMCell(input_dim * 3)
    def make_cell():
        return tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True,
                                       initializer=tf.contrib.layers.xavier_initializer())

    # The LSTM state as a Variable initialized to zeroes
    #lstm_state1 = (tf.Variable(tf.zeros([1, input_dim * 3])),) * 2

    # Connect the input layer and initial LSTM state to the LSTM cell
    #lstm_output1, lstm_state_output1 = lstm_layer1(input_layer, lstm_state1, scope='LSTM1')

    # The LSTM state will get updated
    #lstm_update_op1 = tf.assign(lstm_state1[0], lstm_state_output1[0])
    #lstm_update_op2 = tf.assign(lstm_state1[1], lstm_state_output1[1])

    stack = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(num_layers)])

    lstm_output1, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32, parallel_iterations=1024)
    lstm_output1 = tf.reshape(lstm_output1, [-1, num_features])

    ##The Regression-Output Layer1
    # The Weights and Biases matrices first
    output_W1 = tf.Variable(tf.truncated_normal([num_features, num_features]))
    output_b1 = tf.Variable(tf.zeros([num_features]))
    # Compute the output
    final_output = tf.matmul(lstm_output1, output_W1) + output_b1
    final_output = tf.reshape(final_output, [batch_size, -1, num_features])

    ##Input for correct output (for training)
    targets = tf.placeholder(tf.float32, [batch_size, None, num_features])

    ##Calculate the Sum-of-Squares Error
    error = tf.reduce_sum(tf.pow(final_output - targets, 2))

    ##The Optimizer
    # Adam works best
    train_step = tf.train.AdamOptimizer(0.0006).minimize(error)
    train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(error)

    return inputs, targets, seq_len, input_dropout_keep, output_dropout_keep, state_dropout_keep, \
           error, \
           train_step, learning_rate, final_output

def get_model2():
    ##The Input Layer as a Placeholder
    # Since we will provide data sequentially, the 'batch size'
    # is 1.
    input_layer = tf.placeholder(tf.float32, [1, 22, input_dim * 3])

    # input_layer = tf.placeholder(tf.float32, [1, input_dim * 3])

    ##The LSTM Layer-1
    # The LSTM Cell initialization
    # lstm_layer1 = tf.contrib.rnn.BasicLSTMCell(input_dim * 3)
    def make_cell():
        return tf.contrib.rnn.LSTMCell(2 * input_dim * 3, state_is_tuple=True,
                                       initializer=tf.contrib.layers.xavier_initializer())

    # The LSTM state as a Variable initialized to zeroes
    # lstm_state1 = (tf.Variable(tf.zeros([1, input_dim * 3])),) * 2

    # Connect the input layer and initial LSTM state to the LSTM cell
    # lstm_output1, lstm_state_output1 = lstm_layer1(input_layer, lstm_state1, scope='LSTM1')

    # The LSTM state will get updated
    # lstm_update_op1 = tf.assign(lstm_state1[0], lstm_state_output1[0])
    # lstm_update_op2 = tf.assign(lstm_state1[1], lstm_state_output1[1])

    stack = tf.contrib.rnn.MultiRNNCell(
        [make_cell() for _ in range(1)])

    lstm_output1, _ = tf.nn.dynamic_rnn(stack, input_layer, [22], dtype=tf.float32, parallel_iterations=1024)
    lstm_output1 = tf.reshape(lstm_output1, [-1, input_dim * 3])

    ##The Regression-Output Layer1
    # The Weights and Biases matrices first
    output_W1 = tf.Variable(tf.truncated_normal([input_dim * 3, input_dim]))
    output_b1 = tf.Variable(tf.zeros([input_dim]))
    # Compute the output
    final_output = tf.matmul(lstm_output1, output_W1) + output_b1

    ##Input for correct output (for training)
    correct_output = tf.placeholder(tf.float32, [1, input_dim])

    ##Calculate the Sum-of-Squares Error
    error = tf.reduce_sum(tf.pow(final_output - correct_output, 2))

    ##The Optimizer
    # Adam works best
    train_step = tf.train.AdamOptimizer(0.0006).minimize(error)
    train_step = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(error)
