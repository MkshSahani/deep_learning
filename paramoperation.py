from operation import *


class ParamOperation(Operation):

    def __init__(self, param):
        super().__init__()

        self.param = param

    def backward(self, output_grad):

        # call the self._input_grad
        # call the self._param_grad
        print(self.output)
        print(output_grad)
        assert_same_shape(self.output, output_grad)

        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        # print("=== this is self.param")
        # print(self.param)
        # print("=== this is self.param_grad")
        # print(type(self))
        # print(self.param_grad)
        assert_same_shape(self.param, self.param_grad)
        # print("== = self input _grand")
        # print(self.input_grad)
        return self.input_grad

    def _param_grad(self, output_grad):
        # every subclass of ParamOperation should implement

        raise NotImplementedError()
