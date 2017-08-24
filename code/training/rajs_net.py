import os, tflearn, tensorflow as tf, numpy as np
from feature_extraction import extraction

def create_model(data_shape, learning_rate, classes):
    net = tflearn.input_data(data_shape)
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
    return tflearn.DNN(net, tensorboard_verbose=0)

def batch_generator(data, batch_size):
    alphabet = set()
    for _, _, text in data:
        #if not all(x.isspace() or x.isalpha() for x in text):
        #    continue
        alphabet.update(text.lower())
    alphabet = sorted(list(alphabet))

    samples = []
    labels = []

    mx = max([len(d[1][0]) for d in data])
    while True:
        for i in range(len(data)):
            dt = np.pad(data[i][1], ((0, 0), (0, mx - len(data[i][1][0]))), mode='constant', constant_values=0)
            samples.append(np.array(dt))
            lbl = [np.eye(len(alphabet))[alphabet.index(c)] for c in data[i][2].lower()]
            lbl = np.pad(lbl, )
            labels.append(np.array(lbl))
            if len(samples) >= batch_size:
                yield samples, labels, alphabet, mx

def train(path):
    learning_rate = 0.0001
    training_iters = 300000  # steps
    batch_size = 32

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
            if i >= batch_size:
                break
            data.append((os.path.join(root, file), extraction.get_features_vector(os.path.join(root, file)), fname_to_text[file[:-4]]))
            i += 1

    width = 20  # mfcc features
    height = 80  # (max) length of utterance
    classes = 10  # digits
    print("Generating batches")

    batches = batch_generator(data, batch_size)
    X, Y, alphabet, height = next(batches)
    trainX, trainY = X, Y
    testX, testY = X, Y  # overfit for now

    ### add this "fix" for tensorflow version errors
    col = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for x in col:
        tf.add_to_collection(tf.GraphKeys.VARIABLES, x)

    model = create_model([None, width, height], learning_rate, len(alphabet))

    for _ in range(training_iters):
        model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True,
                  batch_size=batch_size)
        #_y = model.predict(X)
    model.save("tflearn.lstm.model")
    #print(_y)
    #print(y)