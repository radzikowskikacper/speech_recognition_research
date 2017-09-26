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

    #from language_model import training
    #training.main2()
    #os._exit(0)

    #from recognition.ctc import testing
    #testing.test('../data/umeerj/checkpoints/1_5000_1000_3_28_0.0001_0.9_70000_1_0.2_1_1_0.8_0.1_3_1/2017-09-17 11:58:52.824927/')
    #os._exit(0)

    from recognition.ctc import training
    if len(arguments) > 1:
        training.train(sys.argv[1:])
    else:
        training.train(['default', 500, 50, 1, 10, 0.005, 0.9, 1, 1, 1, 1,
                        '../data/umeerj/checkpoints/0_5000_500_2_40_0.0001_0.9_1_0.2_1_1/2017-09-22 10:22:52/'
                        ])