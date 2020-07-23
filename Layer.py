# Layer.py

from utils import *
from paramoperation import *
# Layer class 2nd class of abstraction


class Layer:

    def __init__(self, neurons):
        self.neurons = neurons
        self.first = True
        self.params = []  # List of parameter.
        self.operations = []  # List of operation
        self.param_grad = []  # List of param_gradients

    def _setup_layer(self, num_in):
        '''
            correct series of opeartion in this 
            _setup_layer function. 
        '''

        raise NotImplementedError()

    def forward(self, input_):

        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for operation in self.operations:
            input_ = operation.forward(input_)

        self.output = input_

        return self.output

    def backward(self, output_grad):
        # print("==output grad")
        # print(output_grad)
        # print("===self. output")
        # print(self.output)
        assert_same_shape(self.output, output_grad)
        # print("=== same shape")
        for opeartion in reversed(self.operations):
            output_grad = opeartion.backward(output_grad)
        input_grad = output_grad

        self._param_grads()

        return input_grad

    def _param_grads(self):
        self.param_grad = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grad.append(operation.param_grad)

    def _params(self):

        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param)
