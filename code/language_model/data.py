import numpy as np

def load_corpora(fname):
    with open(fname) as f:
        return f.read()

def batch_generator(lines, batch_size, num_seq):
    chars_per_batch = num_seq * batch_size
    num_batches = len(lines) // chars_per_batch

    data = lines[:num_batches * chars_per_batch]

    # Reshape into n_seqs rows
    data = data.reshape((batch_size, -1))

    for n in range(0, data.shape[1], num_seq):
        # The features
        x = data[:, n:n + num_seq]
        # The targets, shifted by one
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield x, y