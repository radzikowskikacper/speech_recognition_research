import tensorflow as tf
import time
import numpy as np
from . import model, data
from utils import utils

def train(arguments):
    gpu = arguments[0]
    batch_size = arguments[1]        # Sequences per batch
    num_steps = arguments[2]         # Number of sequence steps per batch
    lstm_size = arguments[3]         # Size of hidden layers in LSTMs
    num_layers = arguments[4]          # Number of LSTM layers
    learning_rate = arguments[5]   # Learning rate
    keep_prob = arguments[6]
    num_epochs = arguments[7]

    with open('anna.txt', 'r') as f:
        text=f.read()
    vocab = set(text)
    vocab_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

    batches = data.get_batches(encoded, 10, 50)
    x, y = next(batches)

    initial_state, inputs, targets, keep_probs, losss, final_state, optimizer, _, logits2, _ = model.create_model(len(vocab), batch_size=batch_size, num_steps=num_steps,
                    lstm_size=lstm_size, num_layers=num_layers,
                    learning_rate=learning_rate)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        for epoch in range(num_epochs):
            # Train network
            new_state = sess.run(initial_state)
            lowest_val_error = 2
            for x, y in data.get_batches(encoded, batch_size, num_steps):
                t0 = time.time()
                feed = {inputs: x,
                        targets: y,
                        keep_probs: keep_prob,
                        initial_state: new_state}
                batch_loss, new_state, _ = sess.run([losss,
                                                     final_state,
                                                     optimizer],
                                                    feed_dict=feed)

                dur = time.time() - t0
                log = "E: {}/{}, Tr_loss: {:.3f}, Tr_err: {:.1f}%, Val_loss: {:.3f}, Val_err: {:.1f}%, time: {:.2f} s " \
                      "- - - GPU: {}, H: {}, L: {}, BS: {}, LR: {}, M: {}, Dr-keep: {} / {} / {} / {}, "
                print(log.format(epoch, num_epochs, ))

                print('Epoch: {}... '.format(epoch),
                      'Training loss: {:.4f}... '.format(batch_loss),
                      '{:.4f} sec/batch'.format((end - start)))

                print('Epoch: {} ')
                if error < lowest_val_error:
                    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(lowest_val_error, lstm_size))
                    with open('{}/{}'.format(model_folder_name, params_fname), 'w+') as f:
                        f.write('E: {}\n'.format(curr_epoch))
                        f.write('Tr_err: {}\nVal_err: {}\n'.format(train_ler, val_ler))
                        f.write('{} samples without padding\n{} trainable parameters\n'.format(num_samples,
                                                                                               trainable_parameters))
                    lowest_val_error = val_ler

            #break

        saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))