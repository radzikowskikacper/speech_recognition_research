import librosa, os, wave, contextlib

def get_features_vector_list(path):
    result = []
    for root, dirs, files in os.walk(path):
        for file in files:
            result.append((os.path.join(root, file), get_features_vector(os.pardir.join(root, file))))

def get_features_vector(fname):
    #return fname
    return get_mfcc(fname)

def __wav_length(path):
    with contextlib.closing(wave.open(path, 'r')) as f:
        return f.getnframes() / float(f.getframerate())

def get_mfcc(fname):
    tm = __wav_length(fname)
    if tm == 0.0:
        raise Exception("File corrupted")

    wave, sr = librosa.load(fname)
    return librosa.feature.mfcc(wave, sr)