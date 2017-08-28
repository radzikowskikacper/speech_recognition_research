#from data_handler.handler import handler
import os, sys
#from testing import demo

#h = handler()
path = os.path.expanduser('~')

if __name__ == '__main__':
    '''
    from feature_extraction import speaker_dependency
    speaker_dependency.get_speaker_dependent_features()
    '''

    #from testing import examples
    #examples.sim()

    #demo.do_demo()
    #from feature_extraction import extraction
    #a = extraction.get_features_vector("/media/kapi/0999786741FF823E/research_data/umeerj/ume-erj/wav/JE/DOS/F01/S6_001.wav")

    #from training import rajs_net
    #rajs_net.train('../data/umeerj/ume-erj/')

    from training.normal_s2s_encdec import s2s_encdec
    s2s_encdec.demo(sys.argv[1:])
    #from training.normal_s2s_encdec import original
    #from training.normal_ctc import ctc_tensorflow_example


    '''
    h.loader.load_data('{}/projects/research/phd/data/ume-erj'.format(path))

    h.formatter.transformData_Kaldi_easier_tut('{}/projects/research/kaldi'.format(path),
                                     '{}/projects/research/phd/data/kaldi'.format(path),
                                     [])

    h.formatter.prepare_training_script_Kaldi('{}/projects/research/kaldi'.format(path),
                                     '{}/projects/research/phd/data/kaldi'.format(path),
                                    '{}/projects/research/phd/data/kaldi/trainer.sh'.format(path))
    '''