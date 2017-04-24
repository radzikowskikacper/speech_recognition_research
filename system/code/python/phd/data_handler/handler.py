from .loaders.erj import erj_loader
from .formatters.kaldi import *


class handler:
    __data = data()
    loader = None
    formatter = None

    def __init__(self):
        self.loader = erj_loader(self.__data)
        self.formatter = kaldi_formatter(self.__data)
