from data_handler.handler import handler
import os


h = handler()
path = os.path.expanduser('~')

if __name__ == '__main__':
    h.loader.load_data('{}/projects/research/phd/data/ume-erj'.format(path))

    h.formatter.transformData_Kaldi_easier_tut('{}/projects/research/kaldi'.format(path),
                                     '{}/projects/research/phd/data/kaldi'.format(path),
                                     [])

    h.formatter.prepare_training_script_Kaldi('{}/projects/research/kaldi'.format(path),
                                     '{}/projects/research/phd/data/kaldi'.format(path),
                                    '{}/projects/research/phd/data/kaldi/trainer.sh'.format(path))
