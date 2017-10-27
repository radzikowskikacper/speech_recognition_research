from collections import defaultdict
import numpy as np

def pad_data2(data, padder):
    max_length = max([d.shape[0] for d in data])

    for i in range(len(data)):
        data[i] = np.pad(data[i], ((0, max_length - data[i].shape[0]), (0, 0)), mode='constant', constant_values=padder)

    return data

def batch_generator(samples, targets, batch_size, samples_mean, samples_std, target_parser = None, mode ='training'):
    batches_num = 4
    for batch_i in range(len(samples) // batch_size):
        batch_start = batch_size * batch_i
        rsamples = samples[batch_start:batch_start + batch_size]
        rsamples = pad_data2(rsamples, 0)
        rsamples = (rsamples - samples_mean) / (samples_std)# - samples_min)
        labels = targets[batch_start:batch_start + batch_size]
        samples_lengths = [s.shape[0] for s in rsamples]
        if target_parser:
            labels = target_parser(labels)
        yield rsamples, labels, samples_lengths, targets[batch_start:batch_start + batch_size]
