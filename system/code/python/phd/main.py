from data_handler.handler import handler
from collections import defaultdict

h = handler()
path = '/home/kradziko'
if __name__ == '__main__':
    h.load_data('{}/projects/research/phd/data'.format(path))

    h.transformData_Kaldi('{}/projects/research/phd/data/kaldi'.format(path), ['DOS-F01'])

    pass