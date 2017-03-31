from data_handler.handler import handler
import os

h = handler()
path = os.path.expanduser('~')
if __name__ == '__main__':
    h.load_data('{}/projects/research/phd/data'.format(path))

    h.transformData_Kaldi('{}/projects/research/phd/data/kaldi'.format(path), ['DOS-F01'])

    pass