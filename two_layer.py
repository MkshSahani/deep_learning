import numpy as np 


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    

def forward_pass(input_, output_, weights):
    M1 = np.dot(input_, weights['W1'])
    N1 = M1 + weights['B1']
    O1 = sigmoid(N1)
    M2 = np.dot(O1, weights['W2'])
    P = M2 + weights['B2']
    loss = np.mean(np.power(output_ - P, 2))

    forward_info = {}
    forward_info['X'] = input_
    forward_info['Y'] = output_ 
    forward_info['M1'] = M1
    forward_info['N1'] = N1
    forward_info['O1'] = O1
    forward_info['M2'] = M2
    forward_info['P'] = P
    return forward_info
    

def loss_gradients(forward_info, weights):

    dLdP = -2 * (forward_info['Y'] - forward_info['P']) # matrix with one element or shoud i a scalar. 
    dPdM2 = np.ones_like(forward_info['M2']) 
    dPdB2 = np.ones_like(forward_info['B2'])
    dM2dW2 = np.transpose(forward_info['O1'], (1, 0))
    dO1dN1 = sigmoid(forward_info['N1']) * (1 - sigmoid(forward_info['N1']))
    dN1dB1 = np.ones_like(forward_info['B1'])
    dN1dM1 = np.ones_like(forward_info['W1'])
    dM1dW1 = np.transpose(forward_info['X'], (1, 0))
    
    dLdB2 = (dLdP * dPdB2).sum(axis=0) # matrix 
    
    dLdM2 = dLdP * dPdM2  # matrix 
    
    dLdO1 = np.dot(dLdM2, dM2dO1)
    
    dLdN1 = dLdO1 * dO1dN1

    dLdB1 = (dLdN1 * dN1dB1).sum(axis=0)
    
    dLdW2 = np.dot(dM2dW2, dLdM2)

    

    loss_gradients = {}
    loss_gradients['B1'] = dLdB1
    loss_gradients['B2'] = dLdB2
    
    loss_gradients['W1'] = None
    loss_gradients['W2'] = None
    
    return loss_gradients 



    
