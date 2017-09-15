import librosa, os, wave, contextlib
import scipy.io.wavfile as wav
from python_speech_features import mfcc

def __wav_length(path):
    with contextlib.closing(wave.open(path, 'r')) as f:
        return f.getnframes() / float(f.getframerate())

def get_librosa_mfcc(fname):
    tm = __wav_length(fname)
    if tm == 0.0:
        raise Exception("File corrupted")

    wave, sr = librosa.load(fname, sr=None)
    return librosa.feature.mfcc(wave, sr)

def get_other_mfcc(fname):
    tm = __wav_length(fname)
    if tm == 0.0:
        raise Exception("File corrupted")

    fs, audio = wav.read(fname)
    return mfcc(audio, samplerate=fs)
