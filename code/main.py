import os
import sys

if __name__ == '__main__':
    arguments = sys.argv
    if len(arguments) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments[1])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    from preprocessing.loading.ctc_preprocessing import preprocess_data
    preprocess_data('../data/umeerj/ume-erj/', '../data/umeerj/data2.dat')
    os._exit(0)
    #from recognition.wavenet import train
    #train.train()

    #os._exit(0)
    from recognition.ctc import ctc_tf
    if len(arguments) > 1:
        ctc_tf.train(sys.argv[1:])
    else:
        ctc_tf.train(['default', 500, 500, 2, 1, 0.005, 0.9, 10, 1, 1, 1, 1, 0.5, 0, 3, 1])