# Loss.py

from utils import *


class Loss:

    def __init__(self):
        pass

    def forward(self, prediction, target):
        # calculate the loss value.
        assert_same_shape(prediction, target)
        self.prediction = prediction
        self.target = target
        loss_value = self._output()
        return loss_value

    def backward(self):
        self.input_grad = self._input_grad()
        assert_same_shape(self.prediction, self.input_grad)

        return self.input_grad

    def _output(self):
        # every subclass of Loss should have it's own output function
        raise NotImplementedError()

    def _input_grad(self):
        # every subclass of Loss should have it's own input_grad
        raise NotImplementedError()
