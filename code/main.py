from data_handler.handler import handler
import os
from testing import demo

h = handler()
path = os.path.expanduser('~')

if __name__ == '__main__':
    '''
    from feature_extraction import speaker_dependency
    speaker_dependency.get_speaker_dependent_features()
    '''

    #from testing import examples
    #examples.sim()

    demo.do_demo()
    
    '''
    h.loader.load_data('{}/projects/research/phd/data/ume-erj'.format(path))

    h.formatter.transformData_Kaldi_easier_tut('{}/projects/research/kaldi'.format(path),
                                     '{}/projects/research/phd/data/kaldi'.format(path),
                                     [])

    h.formatter.prepare_training_script_Kaldi('{}/projects/research/kaldi'.format(path),
                                     '{}/projects/research/phd/data/kaldi'.format(path),
                                    '{}/projects/research/phd/data/kaldi/trainer.sh'.format(path))
    '''