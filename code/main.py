import os
import sys

if __name__ == '__main__':
    arguments = sys.argv
    if len(arguments) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments[1])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    from preprocessing.loading import ctc_targeted
    data = ctc_targeted.load_data('../data/umeerj/ume-erj/')
    ctc_targeted.save_data_to_file(data, '../data/umeerj/data_both_mfcc_2.dat')
    os._exit(0)

    #from training.ctc import ctc_tensorflow_multidata_example

    from recognition.ctc import ctc_tf
    if len(arguments) > 1:
        ctc_tf.train(sys.argv[1:])
    else:
        ctc_tf.train(['default', 500, 750, 3, 10, 0.005, 0.9, 10, 0.5, 0.5, 0.5, 1, 0.8, 0.1, 3, 1])