# Linear Regression .

# B = B0 + B1 * X1 + B2*X2 + B3*X3 + ... + BNXN 

import numpy as np
import matplotlib.pyplot as plt 
###################  linear regression forward pass #################

# function name : forward_pass 
## weights => dict parameter - 'W' : weights and 'B' : bais
## X_batch : x input => measured
## Y_batch : y output => mesured 
## returns the tuple of the loss ad 

# just do product the input and output and find the 
def forward_linear_regression(x_batch, y_batch, weights):
    # linear regression forward pass return dictionary forward_pass = {} and loss value
    # forward_pass contain the ,measured input  and output,weights result without base added,
    # predicted value.
    N = np.dot(x_batch, weights['W'])
    P = N + weights['B']

    loss = np.mean(np.power(y_batch - P, 2))

    forward_info = {}
    forward_info['X'] = x_batch 
    forward_info['N'] = N 
    forward_info['P'] = P 
    forward_info['Y'] = y_batch 

    return forward_info, loss 


####################   linear regressiono forward pass ################

# function name : loss gradient 
## forward pass : contain the data from the forward linear regression function 
## weights : contain the data of the weights 
## return data dictionary contain loss 

def loss_gradient(forward_info, weights):
    batch_size = forward_info['X']
    dLdP = -2*(forward_info['Y'] - forward_info['P'])
    dPdN = np.ones_like(forward_info['N'])
    dPdB = np.ones_like(weights['B']) 

    dLdN = dLdP * dPdN 

    dNdW = np.transpose(forward_info['X'], (1, 0))

    dLdW = np.dot(dNdW, dLdN)

    dLdB = (dLdP * dPdB).sum(axis = 0)

    loss_gradients = {}
    loss_gradients['W'] = dLdW
    loss_gradients['B'] = dLdB 

    return loss_gradients 


####################  train function #########################
def train(X_train, Y_train, learning_rate = 0.001, batch_size = 23, return_weight = True, sedd = 80718, epoch = 2000):
    data = X_train.shape
    weights = {}
    weights['W'] = np.ones((data[1], 1))
    weights['B'] = np.ones((1, data[1]))
    for i in range(0, epoch):
        forward_info,loss =  forward_linear_regression(X_train, Y_train,weights)
        loss_grads = loss_gradient(forward_info, weights)
        # print(loss_grads['W'])
        # time.sleep(0.1)
        for key in weights.keys():
            weights[key] -= learning_rate*loss_grads[key]

    if return_weight == True:
        print("this is weights")
        print(weights['W'])
        print("weight matrix")
        return weights

    
################# predict function ###########################
def predict(X_input, weights):
    # print(X_input)
    N = np.dot(X_input, weights['W'])
    
    return N + weights['B']

if __name__ == '__main__':
    input_list = []
    for i in range(0, 10):
        ma = [i]
        input_list.append(ma)
    output_list = []
    for i in range(0, 10):
        if i % 2 == 0:
            ma = [3*i+5]
        else:
            ma = [3*i - 5]
        output_list.append(ma)
    print(input_list)
    print(output_list)
    x_train = np.array(input_list)
    y_train = np.array(output_list)
    print(y_train.shape)
    train_info = train(x_train, y_train)
    input_data = []
    for i in range(0, 10):
        ma = [i]
        input_data.append(ma)
    x_input = np.array([input_data])
    do = predict(x_input, train_info)
    print(do[0,0:,0])
    print(x_train.flatten())
    print(y_train.flatten())
    plt.scatter(x_train.flatten(), y_train.flatten(), marker = 'o')
    plt.plot(x_input.flatten(), do[0,0:,0], 'b')
    plt.show()

    
