from data_handler.handler2 import handler2
import os

h = handler2()
path = os.path.expanduser('~')

if __name__ == '__main__':
    h.load_data('{}/projects/research/phd/data/ume-erj'.format(path))

    h.transformData_Kaldi_easier_tut('{}/projects/research/kaldi'.format(path),
                                     '{}/projects/research/phd/data/kaldi2'.format(path),
                                     [])

    pass