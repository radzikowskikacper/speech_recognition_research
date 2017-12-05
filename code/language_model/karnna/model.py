import tensorflow as tf

def create_model(num_classes, batch_size=64, num_steps=50,
                 lstm_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, sampling=False):

    # When we're using this network for sampling later, we'll be passing in
    # one character at a time, so providing an option for that
    if sampling == True:
        batch_size, num_steps = 1, 1
    else:
        batch_size, num_steps = batch_size, num_steps

    tf.reset_default_graph()

    # Build the input placeholder tensors
    # Declare placeholders we'll feed into the graph
    inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')

    # Keep probability placeholder for drop out layers
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Build the LSTM cell
    ### Build the LSTM Cell

    def build_cell(lstm_size, keep_prob):
        # Use a basic LSTM cell
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)

        # Add dropout to the cell
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop

    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)

    ### Run the data through the RNN layers
    # First, one-hot encode the input tokens
    x_one_hot = tf.one_hot(inputs, num_classes)

    # Run each sequence step through the RNN and collect the outputs
    outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state)
    final_state = state

    # Get softmax predictions and logits
    # Reshape output so it's a bunch of rows, one row for each step for each sequence.
    # That is, the shape should be batch_size*num_steps rows by lstm_size columns
    seq_output = tf.concat(outputs, axis=1)
    x = tf.reshape(seq_output, [-1, lstm_size])

    # Connect the RNN outputs to a softmax layer
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal((lstm_size, num_classes), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(num_classes))

    # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
    # of rows of logit outputs, one for each step and sequence
    logits = tf.matmul(x, softmax_w) + softmax_b

    # Use softmax to get the probabilities for predicted characters
    out = tf.nn.softmax(logits, name='predictions')

    logits2 = tf.reshape(logits, [batch_size, -1, num_classes])
    probabilities = tf.nn.softmax(logits2)
    mindex = tf.argmax(logits2, axis=2)
    i = tf.constant(0)
    prod = tf.constant(1, dtype=tf.float32)
    while_condition = lambda i, prod: tf.less(i, num_steps)
    def body(i, prod):
        return [tf.add(i, 1), tf.multiply(prod, probabilities[0, i, tf.cast(mindex[0, i], tf.int32)])]
    score = tf.while_loop(while_condition, body, [i, prod])[1]

    # Loss and optimizer (with gradient clipping)
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())

    # Softmax cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)

    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    return initial_state, inputs, targets, keep_prob, loss, final_state, optimizer, out, logits2, x_one_hot