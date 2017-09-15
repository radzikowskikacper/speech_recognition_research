import os
import sys

if __name__ == '__main__':
    arguments = sys.argv
    if len(arguments) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments[1])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    import recognition.wavenet.train

    os._exit(0)
    from recognition.ctc import ctc_tf
    if len(arguments) > 1:
        ctc_tf.train(sys.argv[1:])
    else:
        ctc_tf.train(['default', 500, 50, 1, 1, 0.005, 0.9, 2, 1, 1, 1, 1, 0.5, 0, 3, 1])