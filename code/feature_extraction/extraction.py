def get_features_vector_list(file_name):
    maybe_download(source, DATA_DIR)
    if target == Target.speaker: speakers = get_speakers()
    batch_features = []
    labels = []
    files = os.listdir(path)
    while True:
        print("loaded batch of %d files" % len(files))
        shuffle(files)
        for wav in files:
            if not wav.endswith(".wav"): continue
            wave, sr = librosa.load(path + wav, mono=True)
            if target == Target.speaker:
                label = one_hot_from_item(speaker(wav), speakers)
            elif target == Target.digits:
                label = dense_to_one_hot(int(wav[0]), 10)
            elif target == Target.first_letter:
                label = dense_to_one_hot((ord(wav[0]) - 48) % 32, 32)
            else:
                raise Exception("todo : labels for Target!")
            labels.append(label)
            mfcc = librosa.feature.mfcc(wave, sr)
            # print(np.array(mfcc).shape)
            mfcc = np.pad(mfcc, ((0, 0), (0, 80 - len(mfcc[0]))), mode='constant', constant_values=0)
            batch_features.append(np.array(mfcc))
            if len(batch_features) >= batch_size:
                # print(np.array(batch_features).shape)
                # yield np.array(batch_features), labels
                yield batch_features, labels  # basic_rnn_seq2seq inputs must be a sequence
                batch_features = []  # Reset for next batch
                labels = []