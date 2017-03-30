from data_handler.handler import handler
from collections import defaultdict

h = handler()

if __name__ == '__main__':
    h.load_data('/home/kapi/projects/research/phd/data')

    h.transformData_Kaldi('/home/kapi/projects/research/phd/data/kaldi', ['DOS-F01'])

    pass