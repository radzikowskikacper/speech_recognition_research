import tensorflow as tf
import time
import numpy as np
from . import model, data

batch_size = 5        # Sequences per batch
num_steps = 1000         # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
num_layers = 1          # Number of LSTM layers
learning_rate = 0.001   # Learning rate
keep_prob = 0.5

epochs = 20
# Save every N iterations
save_every_n = 200

with open('anna.txt', 'r') as f:
    text=f.read()
vocab = set(text)
vocab_to_int = {c: i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

batches = data.get_batches(encoded, 10, 50)
x, y = next(batches)

initial_state, inputs, targets, keep_probs, losss, final_state, optimizer = model.create_model(len(vocab), batch_size=batch_size, num_steps=num_steps,
                lstm_size=lstm_size, num_layers=num_layers,
                learning_rate=learning_rate)

saver = tf.train.Saver(max_to_keep=100)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    # Use the line below to load a checkpoint and resume training
    # saver.restore(sess, 'checkpoints/______.ckpt')
    counter = 0
    for e in range(epochs):
        # Train network
        new_state = sess.run(initial_state)
        loss = 0
        for x, y in data.get_batches(encoded, batch_size, num_steps):
            counter += 1
            start = time.time()
            feed = {inputs: x,
                    targets: y,
                    keep_probs: keep_prob,
                    initial_state: new_state}
            batch_loss, new_state, _ = sess.run([losss,
                                                 final_state,
                                                 optimizer],
                                                feed_dict=feed)

            end = time.time()
            print('Epoch: {}/{}... '.format(e + 1, epochs),
                  'Training Step: {}... '.format(counter),
                  'Training loss: {:.4f}... '.format(batch_loss),
                  '{:.4f} sec/batch'.format((end - start)))

            if (counter % save_every_n == 0):
                saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    samples = [c for c in prime]
    initial_state, inputs, targets, keep_probs, losss, final_state, optimizer, prediction = model.create_model(len(vocab),
                                                                                                   batch_size=1,
                                                                                                   num_steps=1,
                                                                                                   lstm_size=lstm_size,
                                                                                                   num_layers=num_layers,
                                                                                                   learning_rate=learning_rate)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0, 0] = vocab_to_int[c]
            feed = {inputs: x,
                    keep_prob: 1.,
                    initial_state: new_state}
            preds, new_state = sess.run([prediction, final_state],
                                        feed_dict=feed)

        c = pick_top_n(preds, len(vocab))
        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0, 0] = c
            feed = {inputs: x,
                    keep_prob: 1.,
                    initial_state: new_state}
            preds, new_state = sess.run([prediction, final_state],
                                        feed_dict=feed)

            c = pick_top_n(preds, len(vocab))
            samples.append(int_to_vocab[c])

    return ''.join(samples)

tf.train.latest_checkpoint('checkpoints')
checkpoint = tf.train.latest_checkpoint('checkpoints')
samp = sample(checkpoint, 2000, lstm_size, len(vocab), prime="Far")
print(samp)

checkpoint = 'checkpoints/i1200_l512.ckpt'
samp = sample(checkpoint, 1000, lstm_size, len(vocab), prime="Far")
print(samp)