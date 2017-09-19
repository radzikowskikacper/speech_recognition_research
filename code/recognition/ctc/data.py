import numpy as np
import os
import traceback
from collections import defaultdict
from threading import Thread, Lock

from preprocessing.feature_extraction import mfcc
from random import shuffle

def get_alphabet(data):
    alphabet = sorted(list(set([c for text in data for c in text.lower()])))
    return alphabet

def normalize_data(data):
    mx = np.max([np.max(np.array(d[1])) for d in data])
    mn = np.min([np.min(np.array(d[1])) for d in data])
    dt2 = [(d[0], (np.array(d[1]) - mn) / (mx - mn), d[2], d[3]) for i, d in enumerate(data)]
    return dt2

def pad_data(data, char_to_int):
    samples_max_length = max([d[1].shape[0] for d in data])
    labels_max_length = max([len(d[2]) for d in data])

    for i in range(len(data)):
        data[i] = (data[i][0],
                   np.pad(data[i][1], ((0, samples_max_length - data[i][1].shape[0]), (0, 0)), mode='constant', constant_values=0),
                   data[i][2],
                   data[i][3] + [char_to_int['<PAD>']] * (labels_max_length - len(data[i][3])))

    return data

def pad_data2(data, padder):
    max_length = max([d.shape[0] for d in data])

    for i in range(len(data)):
        data[i] = np.pad(data[i], ((0, max_length - data[i].shape[0]), (0, 0)), mode='constant', constant_values=padder)

    return data

batches_saves = defaultdict(list)

def batch_generator(samples, targets, batch_size, samples_mean, samples_std, target_parser = None, mode ='training'):
    if mode in batches_saves:
        for line in batches_saves[mode]:
            yield line
    else:
        for batch_i in range(len(samples) // batch_size):
            batch_start = batch_size * batch_i
            rsamples = samples[batch_start:batch_start + batch_size]
            rsamples = pad_data2(rsamples, 0)
            rsamples = (rsamples - samples_mean) / (samples_std)# - samples_min)
            labels = targets[batch_start:batch_start + batch_size]
#            labels = pad_data2(labels, 0)
            samples_lengths = [s.shape[0] for s in rsamples]
            labels_lengths = [l.shape[0] for l in labels]
            if target_parser:
                labels = target_parser(labels)
            batches_saves[mode].append((rsamples, labels, samples_lengths, labels_lengths))
            yield rsamples, labels, samples_lengths, targets[batch_start:batch_start + batch_size]

def preprocess_data(path, output_path):
    data_dir = os.path.join(path, 'wav', 'JE')
    doc_dir = os.path.join(path, 'doc', 'JEcontent', 'tab')

    print("Transcript preparation")
    data = []
    fname_to_text = {}
    for filename in ['sentence1.tab', 'word1.tab']:
        with open(os.path.join(doc_dir, filename)) as f:
            for current_line in f.readlines():
                parts = current_line.split('.wav')
                fname_to_text[parts[0]] = parts[-1].strip()

    print("Raw data preparation")

    for root, dirs, files in os.walk(data_dir):
        dirs.sort()
        for file in sorted(files):
            data.append([os.path.join(root, file), fname_to_text[file[:-4]]])
    print('Prepared space for {} data entries'.format(len(data)))

    to_delete = []
    lck = Lock()
    def proc(start, end, id):
        for k in range(start, end):
            if k % 200 == 0 and k > 0:
                print('[{}] {} %'.format(id, (k - start) / (end - start) * 100))
            try:
                data[k].extend([np.array(mfcc.get_librosa_mfcc(data[k][0])), #20 mfcc
                               np.array(mfcc.get_other_mfcc(data[k][0])).T]) #13 mfcc
            except:
                print(data[k][0])
                traceback.print_exc()
                try:
                    lck.acquire()
                    to_delete.append(k)
                finally:
                    lck.release()

    procs = 8
    ts = []
    i = len(data)
    for k in range(procs):
        print("{} - {}, {}".format(int(k * i / procs), int((k + 1) * i / procs), i))
        ts.append(Thread(target=proc, args=(int(k * i / procs), int((k + 1) * i / procs), k,)))

    for t in ts:
        t.start()
    for t in ts:
        t.join()

    print('Deleting {} corrupted file data'.format(len(to_delete)))
    for k in sorted(to_delete, reverse=True):
        del data[k]

    print('Saving {} data entries'.format(len(data)))
    save_data_to_file(data, output_path)

    print('Validating save file')
    data2 = load_data_from_file(output_path, [20, 13])

    assert(len(data) == len(data2))
    for d1, d2 in zip(data, data2):
        assert(d1[0] == d2[0])
        assert(d1[1] == d2[1])
        assert(np.allclose(d1[2], d2[2]))
        assert(np.allclose(d1[3], d2[3]))
    print('Saved successfully')

    return data

def save_data_to_file(data, path):
    with open(path, 'w+') as f:
        f.write('{}\n'.format(len(data)))
        for d in data:
            f.write(d[0] + '\n')
            f.write(d[1] + '\n')
            for i in range(2, len(d)):
                if len(d[i].shape) == 2:
                    for feat in d[i]:
                        f.write(' '.join([str(f) for f in feat]) + '\n')
                else:
                    f.write(' '.join([str(f) for f in d[i]]) + '\n')


def load_data_from_file(path, num_features, samples = None):
    data = []
    with open(path) as f:
        if not samples:
            samples = int(f.readline().strip())
        else:
            f.readline()

        for _ in range(samples):
            fname = f.readline().strip()
            original_text = f.readline().strip()

            to_append = [fname, original_text]
            for rng in num_features:
                if rng > 1:
                    to_append.append(np.array([[float(n) for n in f.readline().split()] for _ in range(rng)]))
                else:
                    to_append.append(np.array([float(n) for n in f.readline().split()]))

            data.append(to_append)

    return data

def load_data_sets(directory):
    training_data = load_data_from_file(os.path.join(directory, 'training.txt'), [20, 13, 1])
    testing_data = load_data_from_file(os.path.join(directory, 'testing.txt'), [20, 13, 1])
    validation_data = load_data_from_file(os.path.join(directory, 'validation.txt'), [20, 13, 1])

    SPACE_TOKEN = '<space>'

    for ds in [training_data, testing_data, validation_data]:
        for i in range(len(ds)):
            ds[i][2] = ds[i][2].T
            ds[i][3] = ds[i][3].T

    alphabet = get_alphabet([d[1] for d in testing_data] + [d[1] for d in training_data] + [d[1] for d in validation_data])
    int_to_char = {i: char for i, char in enumerate([SPACE_TOKEN] + alphabet)}
    char_to_int = {char: i for i, char in int_to_char.items()}
    num_classes = len(int_to_char) + 1

    training_data = sorted(training_data, key= lambda x: x[3].shape[0], reverse=True)
    training_inputs = [d[3] for d in training_data]
    training_targets = [d[4] for d in training_data]
    training_inputs_mean = np.sum([np.sum(s) for s in training_inputs]) / np.sum([np.size(s) for s in training_inputs])
    training_inputs_std = np.sqrt(np.sum([np.sum(np.power(s - training_inputs_mean, 2)) for s in training_inputs]) /
                             np.sum([np.size(s) for s in training_inputs]))

    #testing_data = training_data
    testing_data = sorted(testing_data, key= lambda x: x[3].shape[0], reverse=True)
    testing_inputs = [d[3] for d in testing_data]
    testing_targets = [d[4] for d in testing_data]

    #validation_data = training_data
    validation_data = sorted(validation_data, key= lambda x: x[3].shape[0], reverse=True)
    validation_inputs = [d[3] for d in validation_data]
    validation_targets = [d[4] for d in validation_data]
    #validation_targets = np.array([np.array([10, 20, 30, 40, 50]),
    #                               np.array([0] * 1000000),
    #                                        np.array([0] * 1000000),
    #                                                 np.array([0] * 1000000),
    #                                                          np.array([0] * 1000000)])

    return training_data, training_inputs, training_targets, training_inputs_mean, training_inputs_std, validation_data, \
           validation_inputs, validation_targets, testing_data, testing_inputs, testing_targets, int_to_char, num_classes, \
           sum([d[3].shape[0] for d in validation_data] + [d[3].shape[0] for d in testing_data] + [d[3].shape[0] for d in training_data])

def prepare_data_sets(fname, num_examples, training_part, testing_part):
    SPACE_TOKEN = '<space>'

    data = load_data_from_file(fname, [20, 13])#, num_examples)
    print('Loaded {} data rows'.format(len(data)))

    if num_examples == 70000:
        data = sorted(data, key=lambda x: x[3].shape[1], reverse=True)
        data = data[-num_examples:]
        shuffle(data)
    else:
        shuffle(data)
        data = data[-num_examples:]

    print('Taking {} samples'.format(len(data)))
    print('Totally {} samples, without padding'.format(sum([d[3].shape[1] for d in data])))

    alphabet = get_alphabet([d[1] for d in data])
    int_to_char = {i: char for i, char in enumerate([SPACE_TOKEN] + alphabet)}
    char_to_int = {char: i for i, char in int_to_char.items()}
    num_classes = len(int_to_char) + 1

    all_targets = [d[1] for d in data]
    all_targets = [' '.join(t.strip().lower().split(' '))
                     #.replace('.', '').replace('-', '').replace("'", '').replace(':', '').replace(',', '').
                     #   replace('?', '').replace('!', '')
                 for t in all_targets]
    all_targets = [target.replace(' ', '  ') for target in all_targets]
    all_targets = [target.split(' ') for target in all_targets]
    all_targets = [np.hstack([SPACE_TOKEN if x == '' else list(x) for x in target]) for target in all_targets]
    all_targets = [np.asarray([char_to_int[x] for x in target]) for target in all_targets]
    for i in range(len(data)):
        data[i].append(all_targets[i])
        #data[i][2] = data[i][2].T
        #data[i][3] = data[i][3].T

    training_data = data[:int(training_part * len(data))]
    #training_data = sorted(training_data, key= lambda x: x[3].shape[0], reverse=True)
    training_inputs = [d[3] for d in training_data]
    training_targets = [d[4] for d in training_data]
    training_inputs_mean = np.sum([np.sum(s) for s in training_inputs]) / np.sum([np.size(s) for s in training_inputs])
    training_inputs_std = np.sqrt(np.sum([np.sum(np.power(s - training_inputs_mean, 2)) for s in training_inputs]) /
                             np.sum([np.size(s) for s in training_inputs]))

    testing_data = data[int(training_part * len(data)):int((training_part + testing_part) * len(data))]
    #testing_data = training_data
    #testing_data = sorted(testing_data, key= lambda x: x[3].shape[0], reverse=True)
    testing_inputs = [d[3] for d in testing_data]
    testing_targets = [d[4] for d in testing_data]

    validation_data = data[int((training_part + testing_part) * len(data)):]
    #validation_data = training_data
    #validation_data = sorted(validation_data, key= lambda x: x[3].shape[0], reverse=True)
    validation_inputs = [d[3] for d in validation_data]
    validation_targets = [d[4] for d in validation_data]
    #validation_targets = np.array([np.array([10, 20, 30, 40, 50]),
    #                               np.array([0] * 1000000),
    #                                        np.array([0] * 1000000),
    #                                                 np.array([0] * 1000000),
    #                                                          np.array([0] * 1000000)])

    return training_data, training_inputs, training_targets, training_inputs_mean, training_inputs_std, validation_data, \
           validation_inputs, validation_targets, testing_data, testing_inputs, testing_targets, int_to_char, num_classes, sum([d[3].shape[0] for d in data])