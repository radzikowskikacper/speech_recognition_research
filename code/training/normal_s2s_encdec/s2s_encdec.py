import tensorflow as tf, numpy as np, time, os
from tensorflow.python.layers.core import Dense
from data_handler.loaders import tf_loader
from random import shuffle

def get_model_inputs(number_of_features):
    input_data = tf.placeholder(tf.float32, [None, None, number_of_features], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')

    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

    return input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length

def create_encoder(input_data, rnn_size, num_layers, source_sequence_length):
    # Encoder embedding
    #enc_embed_input = tf.contrib.layers.embed_sequence(input_data,40, embedding_size)

    # RNN cell
    def make_cell(rnn_size):
        enc_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return enc_cell

    enc_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])

    enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, input_data, sequence_length=source_sequence_length,
                                              dtype=tf.float32)

    return enc_output, enc_state

# Process the input we'll feed to the decoder
def process_decoder_input(target_data, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input

def create_decoder(char_to_int, embedding_size, num_layers, rnn_size, target_sequence_length, max_target_sequence_length,
                   input_data_encoder_state, input_data_target, batch_size):
    # 1. Decoder Embedding
    target_vocab_size = len(char_to_int)
    embeddings = tf.Variable(tf.random_uniform([target_vocab_size, embedding_size]))
    dec_embed_input = tf.nn.embedding_lookup(embeddings, input_data_target)

    # 2. Construct the decoder cell
    def make_cell(rnn_size):
        dec_cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return dec_cell

    dec_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])

    # 3. Dense layer to translate the decoder's output at each time
    # step into a choice from the target vocabulary
    output_layer = Dense(target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

    # 4. Training Decoder
    with tf.variable_scope("decode"):
        # Helper for the training process. Used by BasicDecoder to read inputs.
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=target_sequence_length,
                                                            time_major=False)

        # Basic decoder
        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           input_data_encoder_state,
                                                           output_layer)

        # Perform dynamic decoding using the decoder
        training_decoder_output = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                    impute_finished=True,
                                                                    maximum_iterations=max_target_sequence_length)
    # 5. Inference Decoder
    # Reuses the same parameters trained by the training process
    with tf.variable_scope("decode", reuse=True):
        start_tokens = tf.tile(tf.constant([char_to_int['<GO>']], dtype=tf.int32), [batch_size],
                               name='start_tokens')

        # Helper for the inference process.
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                    start_tokens,
                                                                    char_to_int['<EOS>'])

        # Basic decoder
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            inference_helper,
                                                            input_data_encoder_state,
                                                            output_layer)

        # Perform dynamic decoding using the decoder
        inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                     impute_finished=True,
                                                                     maximum_iterations=max_target_sequence_length)

    return training_decoder_output, inference_decoder_output

def create_model(input_data, targets, lr, target_sequence_length, max_target_sequence_length, rnn_size, num_layers, char_to_int, source_sequence_length, batch_size, embedding_size):
    # Pass the input data through the encoder. We'll ignore the encoder output, but use the state
    _, enc_state = create_encoder(input_data, rnn_size, num_layers, source_sequence_length)

    # Prepare the target sequences we'll feed to the decoder in training mode
    dec_input = process_decoder_input(targets, char_to_int, batch_size)

    # Pass encoder state and decoder inputs to the decoders
    training_decoder_output, inference_decoder_output = create_decoder(char_to_int,
                                                                       embedding_size,
                                                                       num_layers,
                                                                       rnn_size,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       enc_state,
                                                                       dec_input, batch_size)

    return training_decoder_output, inference_decoder_output

def demo(arguments):
    #'''
    batch_size = int(arguments[0])
    # Number of Epochs
    epochs = int(arguments[1])
    # RNN Size
    rnn_size = int(arguments[2])
    # Number of Layers
    num_layers = int(arguments[3])
    # Learning Rate
    learning_rate = float(arguments[4])
    embedding_size = int(arguments[5])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments[6])
    samples = int(arguments[7])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    display_step = int(arguments[8])  # Check training loss after every 20 batches
    #'''
    data = tf_loader.load_data_from_file('data4.dat', 20, samples)#tf_loader.load_data('../data/umeerj/ume-erj/')
    #data = tf_loader.load_data('../data/umeerj/ume-erj/')
    alphabet = tf_loader.get_alphabet(data)
    tokens = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    int_to_char = {i : char for i, char in enumerate(tokens + alphabet)}
    char_to_int = {char : i for i, char in int_to_char.items()}

    data = [(d[0], d[1], d[2], [char_to_int[char] for char in d[3].lower()]) for d in data]
    #data = tf_loader.normalize_data(data)
    #data = tf_loader.pad_data(data, char_to_int)
    print(data[0][1].shape[1])
    #tf_loader.save_data_to_file(data, 'data4.dat')
    print(len(data))
    print(char_to_int)
    #return

    # Build the graph
    tf.logging.set_verbosity(tf.logging.ERROR)
    train_graph = tf.Graph()
    # Set the graph to default to ensure that it is ready for training
    with train_graph.as_default():
        # Load the model inputs
        input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_model_inputs(data[0][1].shape[1])

        # Create the training and inference logits
        training_decoder_output, inference_decoder_output = create_model(input_data,
                                                                          targets,
                                                                          lr,
                                                                          target_sequence_length,
                                                                          max_target_sequence_length,
                                                                          rnn_size,
                                                                          num_layers, char_to_int, source_sequence_length,
                                                                         batch_size, embedding_size)

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

            accuracy = tf.metrics.accuracy(labels=targets, predictions=inference_logits)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)

    checkpoint = "../data/best_model.ckpt"

    #shuffle(data)
    training_data = data#[:int(0.8 * len(data))]
    testing_data = data[int(0.8 * len(data)):]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.allow_soft_placement=True
    #config.log_device_placement=True
    with tf.Session(graph=train_graph, config = config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for epoch_i in range(1, epochs + 1):
            for batch_i, (source_batch, target_batch, source_lengths, target_lengths) in enumerate(
                    tf_loader.batch_generator(training_data, batch_size, char_to_int)):
                tstart = time.time()
                # Training step
                _, loss, training_accuracy = sess.run(
                    [train_op, cost, accuracy],
                    {input_data: source_batch,
                     targets: target_batch,
                     lr: learning_rate,
                     target_sequence_length: target_lengths,
                    source_sequence_length: source_lengths})

                # Debug message updating us on the status of the training
                if batch_i % display_step == 0:# and batch_i > 0:
                    accs1 = []
                    accs2 = []
                    val_losses = []

                    for j, (testing_source_batch, testing_target_batch, testing_source_lengths, testing_target_lengths) in enumerate(tf_loader.batch_generator(
                        training_data, batch_size, char_to_int, 'testing')):
                        validation_loss, validation_accuracy = sess.run(
                            [cost, accuracy],
                            {input_data: testing_source_batch,
                             targets: testing_target_batch,
                             lr: learning_rate,
                             target_sequence_length: testing_target_lengths,
                             source_sequence_length: testing_source_lengths})
                        accs1.append(validation_accuracy[0])
                        accs2.append(validation_accuracy[1])
                        val_losses.append(validation_loss)

                    print('Epoch {:>3}/{} Batch {:>4}/{}  - '
                          'Training loss: {:>6.3f}  - Training accuracy: {}  - Validation loss: {:>6.3f}  - Validation accuracy: {}  - '
                          'Batch size: {}  - RNN size: {}  - Layers: {}  - LR: {}  - Emb: {}  - Time: {} s'
                          .format(epoch_i,
                                  epochs,
                                  batch_i,
                                  len(data) // batch_size,
                                  loss, training_accuracy,
                                  np.average(np.array(val_losses)), (np.average(np.array(accs1)), np.average(np.array(accs2))),
                                  batch_size, rnn_size, num_layers, learning_rate, embedding_size, time.time() - tstart))
                    #break

        # Save Model
        saver = tf.train.Saver()
        saver.save(sess, checkpoint)
        print('Model Trained and Saved')

    checkpoint = "../data/best_model.ckpt"

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph, config=config) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)

        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')

        for j in range(2):
            # Multiply by batch_size to match the model's input parameters
            answer_logits = sess.run(logits, {input_data: [training_data[j][1]] * batch_size,
                                              target_sequence_length: [1000] * batch_size,
                                              source_sequence_length: [100] * batch_size})[0]
            print(training_data[j][0])
            print(training_data[j][1].shape)
            print(training_data[j][2])
            print(training_data[j][3])
            print('  Word Ids:       {}'.format([i for i in answer_logits if i != char_to_int["<PAD>"]]))
            print('  Response Words: {}'.format(" ".join([int_to_char[i] for i in answer_logits if i != char_to_int["<PAD>"]])))