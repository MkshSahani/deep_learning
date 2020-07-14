# MeanSquaredError.py

from Loss import *
from utils import *
import numpy as np


class MeanSquaredError(Loss):

    def __init__(self):
        super().__init__()

    def _output(self):
        loss = np.sum(np.power(self.prediction - self.target,
                               1 / 2)) / self.prediction.shape[0]

        return loss

    def _input_grad(self):
        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]
