# linear regression version : 1.o.
import numpy as np
import time 
import matplotlib.pyplot as plt 

def forward_linear_regression(x_batch, y_batch, weights):
    # x_bath : ndarrray => matrix input 
    # y_bath : measured output => matrix measured output
    # weights : parameters 

    N = np.dot(x_batch, weights['W'])
    P = N + weights['B']

    loss = np.mean(np.power(y_batch - P, 2))

    forward_info = {}
    forward_info['X'] = x_batch 
    forward_info['N'] = N 
    forward_info['P'] = P 
    forward_info['Y'] = y_batch 

    return forward_info, loss 

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

def train(X_train, Y_train, learning_rate = 0.001, batch_size = 23, return_weight = True, sedd = 80718, epoch = 2000):
    data = X_train.T.shape
    weights = {}
    weights['W'] = np.ones(data)
    weights['B'] = np.ones((1, data[1]))
    for i in range(0, epoch):
        forward_info,loss =  forward_linear_regression(X_train, Y_train,weights)
        loss_grads = loss_gradient(forward_info, weights)
        # print(loss_grads['W'])
        # time.sleep(0.1)
        for key in weights.keys():
            weights[key] -= learning_rate*loss_grads[key]

    if return_weight == True:
        return weights

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
        ma = [3*i+7]
        output_list.append(ma)
    print(input_list)
    print(output_list)
    x_train = np.array(input_list)
    y_train = np.array(output_list)
    print(y_train.shape)
    train_info = train(x_train, y_train)
    input_data = []
    for i in range(11, 40):
        ma = [i]
        input_data.append(ma)
    x_input = np.array([input_data])
    do = predict(x_input, train_info)
    print(do[0,0:,0])
    print(x_train.flatten())
    print(y_train.flatten())
    plt.plot(x_train.flatten(), y_train.flatten(), 'g')
    plt.plot(x_input.flatten(), do[0,0:,0], 'r')
    plt.show()

    
