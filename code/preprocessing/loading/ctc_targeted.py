import numpy as np
import os
import traceback
from collections import defaultdict
from threading import Thread, Lock

from preprocessing.feature_extraction import mfcc


def get_alphabet(data):
    alphabet = sorted(list(set([c for _, _, _, text, _, in data for c in text.lower()])))
    return alphabet

def get_alphabet2(data):
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

def preprocess_data(path):
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

    i = 0
    for root, dirs, files in os.walk(data_dir):
        if i == 10: break
        dirs.sort()
        for file in sorted(files):
            data.append([os.path.join(root, file), fname_to_text[file[:-4]]])
            i += 1
            if i == 10: break

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

    procs = 1
    ts = []
    i = len(data)
    for k in range(procs):
        print("{} - {}, {}".format(int(k * i / procs), int((k + 1) * i / procs), i))
        ts.append(Thread(target=proc, args=(int(k * i / procs), int((k + 1) * i / procs), k,)))

    for t in ts:
        t.start()
    for t in ts:
        t.join()

    for k in sorted(to_delete, reverse=True):
        del data[k]

    return data

def save_data_to_file(data, path):
    with open(path, 'w+') as f:
        f.write('{}\n'.format(len(data)))
        for d in data:
            f.write(d[0] + '\n')
            f.write(d[1] + '\n')
            for i in range(2, len(d)):
                for feat in d[i]:
                    f.write(' '.join([str(f) for f in feat]) + '\n')

def load_data_from_file(path, num_features, samples = None):
    data = []
    with open(path) as f:
        if not samples:
            samples = int(f.readline().strip())

        for _ in range(samples):
            fname = f.readline().strip()
            original_text = f.readline().strip()

            to_append = [fname, original_text]
            for rng in num_features:
                to_append.append(np.array([[float(n) for n in f.readline().split()] for _ in range(rng)]))

            data.append(to_append)

    return data
