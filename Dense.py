# Dense.py

from utils import *
from Layer import *
from sigmoid_block import *
from weightmultiply import *
from biasadd import *
import numpy as np


class Dense(Layer):

    def __init__(self, neurons, activation=SigMoid()):
        super().__init__(neurons)

        self.activation = activation
        self.seed = True 
    def _setup_layer(self, input_):
        # fully connected layer

        if self.seed:
            np.random.seed(self.seed)

        self.params = []

        # Weights

        self.params.append(np.random.randn(input_.shape[1], self.neurons))

        # Bias

        self.params.append(np.random.randn(1, self.neurons))

        # operations

        self.operations = [WeightMultiply(self.params[0]),
                           BiasAdd(self.params[1]),
                           self.activation
                           ]

        return None
