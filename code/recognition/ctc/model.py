import tensorflow as tf

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