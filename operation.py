# operation.py
from utils import *


class Operation:
    # base class for an operation in neural network.
    def __init__(self):
        pass

    def forward(self, input_):
        # store input in self._input instance and call self._output
        self.input_ = input_
        self.output = self._output()
        return self.output

    def backward(self, output_grad):
        # call the self._input_grad() function
        # check that the appropriate shape match.
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)

        return self.input_grad

    def _output(self):
        # output must be define for each operation
        raise NotImplementedError()

    def _input_grad(self, output_grad):
        # the _input_grad method must be define for each opearation .

        raise NotImplementedError()
