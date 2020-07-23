from operation import * 
import numpy as np 

class Linear(Operation):
    '''
    "Identity" activation function
    '''

    def __init__(self):
        '''Pass'''        
        super().__init__()

    def _output(self):
        '''Pass through'''
        return self.input_

    def _input_grad(self, output_grad):
        '''Pass through'''
        return output_grad

        