def pick_top_n(preds, vocab_size, top_n=5):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c

def sample(checkpoint, n_samples, lstm_size, vocab_size, prime="The "):
    samples = [c for c in prime]
    initial_state, inputs, targets, keep_probs, losss, final_state, optimizer, prediction, logits2, xnh = model.create_model(len(vocab),
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
                    keep_probs: 1.,
                    initial_state: new_state}
            l2, preds, new_state, xnhh = sess.run([logits2, prediction, final_state, xnh],
                                        feed_dict=feed)
        print(l2)
        print(x)
        print(xnhh)
        c = pick_top_n(preds, len(vocab))
        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0, 0] = c
            feed = {inputs: x,
                    keep_probs: 1.,
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