# sigmoid_block.py

from operation import *
import numpy as np


class SigMoid(Operation):
    def __init__(self):
        super().__init__()

    def _output(self):
        return 1.0 / (1 + np.exp(-1.0 * self._input))

    def _input_grad(self, output_grad):
        sigmoid_backward = self.output * (1 - self.ouput)
        input_grad = sigmoid_backward * output_grad
        return input_grad
