import tensorflow as tf
from tensorflow.python.layers.core import Dense

def get_model_inputs():
    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')

    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

    return input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length

def create_encoder(input_data, rnn_size, num_layers,
                   source_sequence_length, source_vocab_size,
                   encoding_embedding_size):
    # Encoder embedding
    enc_embed_input = tf.contrib.layers.embed_sequence(input_data, source_vocab_size, encoding_embedding_size)

    # RNN cell
    def make_cell(rnn_size):
        enc_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return enc_cell

    enc_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])

    enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_embed_input, sequence_length=source_sequence_length,
                                              dtype=tf.float32)

    return enc_output, enc_state

def create_decoder(target_letter_to_int, decoding_embedding_size, num_layers, rnn_size,
                   target_sequence_length, max_target_sequence_length, enc_state, dec_input):
    # 1. Decoder Embedding
    target_vocab_size = len(target_letter_to_int)
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    # 2. Construct the decoder cell
    def make_cell(rnn_size):
        dec_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return dec_cell

    dec_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])

    # 3. Dense layer to translate the decoder's output at each time
    # step into a choice from the target vocabulary
    output_layer = Dense(target_vocab_size,
                         kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    # 4. Set up a training decoder and an inference decoder
    # Training Decoder
    with tf.variable_scope("decode"):
        # Helper for the training process. Used by BasicDecoder to read inputs.
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)

        # Basic decoder
        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           enc_state,
                                                           output_layer)

        # Perform dynamic decoding using the decoder
        training_decoder_output = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                    impute_finished=True,
                                                                    maximum_iterations=max_target_sequence_length)
    # 5. Inference Decoder
    # Reuses the same parameters trained by the training process
    with tf.variable_scope("decode", reuse=True):
        start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']], dtype=tf.int32), [batch_size],
                               name='start_tokens')

        # Helper for the inference process.
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                                                                    start_tokens,
                                                                    target_letter_to_int['<EOS>'])

        # Basic decoder
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            inference_helper,
                                                            enc_state,
                                                            output_layer)

        # Perform dynamic decoding using the decoder
        inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                     impute_finished=True,
                                                                     maximum_iterations=max_target_sequence_length)

    return training_decoder_output, inference_decoder_output

def create_model(input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length,
                  source_vocab_size, target_vocab_size, enc_embedding_size, dec_embedding_size, rnn_size, num_layers):
    # Pass the input data through the encoder. We'll ignore the encoder output, but use the state
    _, enc_state = create_encoder(input_data,
                                  rnn_size,
                                  num_layers,
                                  source_sequence_length,
                                  source_vocab_size,
                                  encoding_embedding_size)

    # Prepare the target sequences we'll feed to the decoder in training mode
    dec_input = process_decoder_input(targets, target_letter_to_int, batch_size)

    # Pass encoder state and decoder inputs to the decoders
    training_decoder_output, inference_decoder_output = create_decoder(target_letter_to_int,
                                                                       decoding_embedding_size,
                                                                       num_layers,
                                                                       rnn_size,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       enc_state,
                                                                       dec_input)

    return training_decoder_output, inference_decoder_output

def train():
    # Build the graph
    train_graph = tf.Graph()
    # Set the graph to default to ensure that it is ready for training
    with train_graph.as_default():
        # Load the model inputs
        input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_model_inputs()

        # Create the training and inference logits
        training_decoder_output, inference_decoder_output = create_model(input_data,
                                                                          targets,
                                                                          lr,
                                                                          target_sequence_length,
                                                                          max_target_sequence_length,
                                                                          source_sequence_length,
                                                                          len(source_letter_to_int),
                                                                          len(target_letter_to_int),
                                                                          encoding_embedding_size,
                                                                          decoding_embedding_size,
                                                                          rnn_size,
                                                                          num_layers)

        # Create tensors for the training logits and inference logits
        training_logits = tf.identity(training_decoder_output[0].rnn_output, 'logits')
        inference_logits = tf.identity(inference_decoder_output[0].sample_id, name='predictions')

        # Create the weights for sequence_loss
        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            # Loss function
            cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                targets,
                masks)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
