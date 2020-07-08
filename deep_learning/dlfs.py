import numpy as np

"""
@class name : Operation 
base class for the other function in our netowork. 
"""


class Operation:
    # This is base class for our model.
    # every function here is and operation.

    def __init__(self):
        # default constructor.
        pass

    def forward(self, input_):
        self.input_ = input_
        self.output = self._output()

        return self.ouput

    def backward(self, output_grad):
        self.input_grad = self._input_grad()
        return self.input_grad

    def _output(self):
        raise NotImplementedError()

    def _input_grad(self, output_grad):
        raise NotImplementedError()


"""
@classname :  ParamOperation opeartion with paratmeter. 
"""


class ParamOperation(Operation):

    def __init__(self, param):
        super().__init__()
        self.param = param

    def backward(self, output_grad):
        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        return self.input_grad

    def _param_grad(self, output_grad):
        raise NotImplementedError()


"""
@classname : WeightMultiply opeartion for neural network.
"""


class WeightMultiply(ParamOperation):

    def __init__(self, W):
        super().__init__(W)

    def _output(self):
        # output of the WeightMultiply
        return np.dot(self.input_, self.param)

    def _input_grad(self, output_grad):
        # find the input grand

        return np.dot(output_grad, np.transpose(self.param, (1, 0)))

    def _param_grad(self, output_grad):
        # find the param grand

        return np.dot(np.transpose(self.input_, (1, 0)), output_grad)


"""
@classname : BaisAdd operatoin of neural network. 
"""


class BaisAdd(ParamOperation):

    def __init__(self, B):
        # initialise the Bais
        super().__init__(B)

    def _ouput(self):
        return self.input_ + self.param

    def _input_grad(self, output_grad):
        return output_grad * np.ones_like(self.input_)

    def _param_grad(self, output_grad):
        param_grad = output_grad * np.ones_like(self.param)
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])


"""
@classname : Sigmoid activation function in our neural network. 
"""


class Sigmoid(Operation):

    def __init__(self):
        super().__init__()

    def _output(self):
        # compute output
        return (1 / (1 + np.exp(-1 * self.input_)))

    def _input_grad(self, output_grad):
        # compute input grad
        sigmoid_backward = self.output * (1 - self.output)
        input_grad = sigmoid_backward * output_grad
        return input_grad


"""
@classname : Layer 
"""


class Layer:

    def __init__(self, neurons):
        self.neurons = neurons
        self.first = True
        self.params = []
        self.param_grad = []
        self.operations = []

    def _setup_layer(self, num_in):
        # the _setup_layer function must be implemented for each layer
        raise NotImplementedError()

    def forward(self, input_):
        if self.first:
            self._setup_layer(input_)
            self.first = False

        self.input_ = input_

        for opeartion in self.operations:
            input_ = opeartion.forward(input_)
        self.output = input_
        return self.output

    def backward(self, output_grad):
        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)
        input_grad = output_grad
