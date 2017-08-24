import librosa, os

def get_features_vector_list(path):
    result = []
    for root, dirs, files in os.walk(path):
        for file in files:
            result.append((os.path.join(root, file), get_features_vector(os.pardir.join(root, file))))

def get_features_vector(fname):
    #return fname
    return get_mfcc(fname)

def get_mfcc(fname):
    wave, sr = librosa.load(fname)
    return librosa.feature.mfcc(wave, sr)