import os, sys

if __name__ == '__main__':
    arguments = sys.argv
    if len(arguments) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments[1])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    #from training.normal_ctc import ctc_tensorflow_multidata_example

    from training.normal_ctc import ctc_tensorflow_example
    if len(arguments) > 1:
        ctc_tensorflow_example.train(sys.argv[1:])
    else:
        ctc_tensorflow_example.train(['default', 12, 50, 1, 2, 0.005, 0.9, 10, 1, 1, 1, 1, 0.6, 0.2, 0, 0])