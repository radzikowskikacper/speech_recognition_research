import os
import sys

if __name__ == '__main__':
    arguments = sys.argv
    if len(arguments) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments[1])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    #from preprocessing.loading.ctc_preprocessing import preprocess_data
    #preprocess_data('../data/umeerj/ume-erj/', '../data/umeerj/data2.dat')
    #os._exit(0)
    #from recognition.wavenet import train
    #train.train()

    from language_model import training as lm_training
    lm_training.main2()
    os._exit(0)

    from recognition.ctc import training
    if len(arguments) > 1:
        training.train(sys.argv[1:])
    else:
        training.train(['default', 500, 50, 1, 50, 0.005, 0.9, 1, 1, 1, 1])