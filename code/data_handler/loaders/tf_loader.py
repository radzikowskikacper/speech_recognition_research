import os, numpy as np
from feature_extraction import extraction
from random import shuffle

def get_alphabet(data):
    alphabet = sorted(list(set([c for _, _, text in data for c in text])))
    return alphabet

def pad_data(data, char_to_int):
    samples_max_length = max([d[1].shape[0] for d in data])
    labels_max_length = max([len(d[2]) for d in data])

    for i in range(len(data)):
        data[i] = (data[i][0],
                   np.pad(data[i][1], ((0, samples_max_length - data[i][1].shape[0]), (0, 0)), mode='constant', constant_values=0),
                   data[i][2] + [char_to_int['<PAD>']] * (labels_max_length - len(data[i][2])))

    return data

def batch_generator(data, batch_size, mix = True):
    samples = []
    labels = []

    while True:
        if mix:
            shuffle(data)
        for i in range(len(data)):
            samples.append(data[i][1])
            labels.append(data[i][2])
            if len(samples) >= batch_size:
                samples_lengths = [d.shape[0] for d in samples]
                labels_lengths = [len(l) for l in labels]
                yield samples, labels, samples_lengths, labels_lengths
                samples = []
                labels = []
        #break

def load_data(path):
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
        for file in files:
            if i >= 200:
                break
            data.append((os.path.join(root, file), np.array(extraction.get_features_vector(os.path.join(root, file))).T,
                         fname_to_text[file[:-4]].lower()))
            i += 1

    return data

def save_data_to_file(data, path):
    with open(path, 'w+') as f:
        for d in data:
            f.write(d[0] + '\n')
            for feat in d[1].T:
                f.write(' '.join([str(f) for f in feat]) + '\n')
            f.write(' '.join([str(f) for f in d[2]]) + '\n')