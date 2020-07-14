# weightmultiply.py

from paramoperation import *
import numpy as np


class WeightMultiply(ParamOperation):
    # weight multiplication operation for neural network.

    def __init__(self, W):
        super().__init__(W)

    def _output(self):
        # compute output
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad):
        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad):
        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)
