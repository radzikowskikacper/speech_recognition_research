def get_speaker_dependent_features():
    from python_speech_features import mfcc
    from python_speech_features import logfbank
    import scipy.io.wavfile as wav

    (rate, sig) = wav.read("/home/kradziko/projects/research/phd/data/ume-erj/wav/JE/DOS/F01/S6_002.wav")
    mfcc_feat = mfcc(sig, rate)
    fbank_feat = logfbank(sig, rate)

    print(fbank_feat[1:3, :])

def get_speaker_independent_features():
    pass