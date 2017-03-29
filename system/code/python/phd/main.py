from data_handler.handler import handler
from collections import defaultdict

h = handler()

if __name__ == '__main__':
    h.load_data('/home/kradziko/projects/research/phd/data')

    h.transformData_Kaldi('/home/kradziko/projects/research/phd/data/kaldi', [1,2,3])