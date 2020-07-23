# biasadd.py

from utils import *
from paramoperation import *
import numpy as np


class BiasAdd(ParamOperation):

    def __init__(self, B):
        assert B.shape[0] == 1
        super().__init__(B)

    def _output(self):
        return self.input_ + self.param

    def _input_grad(self, output_grad):
        return np.ones_like(self.input_)*output_grad

    def _param_grad(self, output_grad):
        return np.ones_like(self.param) * output_grad.sum(axis = 0)
