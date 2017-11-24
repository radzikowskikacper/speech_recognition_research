import tensorflow as tf

initial_state = None
inputs = None
targets = None
keep_prob = None
loss = None
final_state = None
optimizer = None

def build_inputs(batch_size, num_steps):
    ''' Define placeholders for inputs, targets, and dropout

        Arguments
        ---------
        batch_size: Batch size, number of sequences per batch
        num_steps: Number of sequence steps in a batch

    '''
    # Declare placeholders we'll feed into the graph
    inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
    targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')

    # Keep probability placeholder for drop out layers
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return inputs, targets, keep_prob

def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    ''' Build LSTM cell.

        Arguments
        ---------
        keep_prob: Scalar tensor (tf.placeholder) for the dropout keep probability
        lstm_size: Size of the hidden layers in the LSTM cells
        num_layers: Number of LSTM layers
        batch_size: Batch size

    '''

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

    return cell, initial_state

def build_output(lstm_output, in_size, out_size):
    ''' Build a softmax layer, return the softmax output and logits.

        Arguments
        ---------

        x: Input tensor
        in_size: Size of the input tensor, for example, size of the LSTM cells
        out_size: Size of this softmax layer

    '''

    # Reshape output so it's a bunch of rows, one row for each step for each sequence.
    # That is, the shape should be batch_size*num_steps rows by lstm_size columns
    seq_output = tf.concat(lstm_output, axis=1)
    x = tf.reshape(seq_output, [-1, in_size])

    # Connect the RNN outputs to a softmax layer
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))

    # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
    # of rows of logit outputs, one for each step and sequence
    logits = tf.matmul(x, softmax_w) + softmax_b

    # Use softmax to get the probabilities for predicted characters
    out = tf.nn.softmax(logits, name='predictions')

    return out, logits

def build_loss(logits, targets, lstm_size, num_classes):
    ''' Calculate the loss from the logits and the targets.

        Arguments
        ---------
        logits: Logits from final fully connected layer
        targets: Targets for supervised learning
        lstm_size: Number of LSTM hidden units
        num_classes: Number of classes in targets

    '''

    # One-hot encode targets and reshape to match logits, one row per batch_size per step
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())

    # Softmax cross entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    return loss

def build_optimizer(loss, learning_rate, grad_clip):
    ''' Build optmizer for training, using gradient clipping.

        Arguments:
        loss: Network loss
        learning_rate: Learning rate for optimizer

    '''

    # Optimizer for training, using gradient clipping to control exploding gradients
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))

    return optimizer

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
    global inputs, initial_state, targets, keep_prob, loss, final_state, optimizer
    inputs, targets, keep_prob = build_inputs(batch_size, num_steps)

    # Build the LSTM cell
    cell, initial_state = build_lstm(lstm_size, num_layers, batch_size, keep_prob)

    ### Run the data through the RNN layers
    # First, one-hot encode the input tokens
    x_one_hot = tf.one_hot(inputs, num_classes)

    # Run each sequence step through the RNN and collect the outputs
    outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state)
    final_state = state

    # Get softmax predictions and logits
    prediction, logits = build_output(outputs, lstm_size, num_classes)

    # Loss and optimizer (with gradient clipping)
    loss = build_loss(logits, targets, lstm_size, num_classes)
    optimizer = build_optimizer(loss, learning_rate, grad_clip)