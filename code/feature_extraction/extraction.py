import librosa, os, wave, contextlib
import scipy.io.wavfile as wav
from python_speech_features import mfcc

def get_features_vector_list(path):
    result = []
    for root, dirs, files in os.walk(path):
        for file in files:
            result.append((os.path.join(root, file), get_features_vector(os.pardir.join(root, file))))

def get_features_vector2(fname):
    #return fname
    return get_librosa_mfcc(fname)

def __wav_length(path):
    with contextlib.closing(wave.open(path, 'r')) as f:
        return f.getnframes() / float(f.getframerate())

def get_librosa_mfcc(fname):
    tm = __wav_length(fname)
    if tm == 0.0:
        raise Exception("File corrupted")

    wave, sr = librosa.load(fname)
    return librosa.feature.mfcc(wave, sr)

def get_other_mfcc(fname):
    tm = __wav_length(fname)
    if tm == 0.0:
        raise Exception("File corrupted")

    fs, audio = wav.read(fname)
    return mfcc(audio, samplerate=fs)
