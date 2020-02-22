

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## forward pass process
def forward_pass(x_batch, y_batch, weights):
    N = np.dot(x_batch, weights['W'])
    P = N +  weights['B']

    loss = np.mean(np.power(y_batch - P, 2))

    forward_info = {}
    forward_info['X'] = x_batch
    forward_info['Y'] = y_batch
    forward_info['N'] = N 
    forward_info['P'] = P 

    return forward_info, loss

## BackWard pass to get the loss due to chage in the weights 
def loss_gradients(forward_info, weights):

    dLdP = -2*(forward_info['Y'] - forward_info['P'])
    dPdB = np.ones_like(weights['B'])
    dPdN = np.ones_like(forward_info['N'])
    dNdW = np.transpose(forward_info['X'], (1, 0))
    dLdN = dLdP*dPdN 
    dLdW = np.dot(dNdW, dLdN)
    dLdB = (dLdP*dPdB).sum(axis = 0)
    loss_gradient = {}
    loss_gradient['W'] = dLdW 
    loss_gradient['B'] = dLdB 
    return loss_gradient

def train(x_train, y_train, learning_rate = 0.001, return_weights = True, epoch = 2000):
    shape_ = x_train.shape
    weights = {}
    weights['W'] = np.random.rand(shape_[1],1)
    weights['B'] = np.random.rand(shape_[1])
    for i in range(0, 2000):
        forward_info,loss = forward_pass(x_train, y_train, weights)
        loss_grad = loss_gradients(forward_info, weights)

        for key in weights.keys():
            weights[key] -= learning_rate * loss_grad[key]
    
    train_info = [weights, return_weights]
    return train_info

def predict(x_input, weights):
    N = np.dot(x_input, weights['W'])
    P = N + weights['B']
    return P


if __name__ == '__main__':
    data_csv = pd.read_csv('data_linear_regression.csv')
    x_input = data_csv.iloc[:,0].values
    plot_input = x_input
    x_input = np.array(x_input)
    x_input = x_input.reshape(x_input.shape[0], 1)
    print(x_input)
    y_output = data_csv.iloc[:,1].values 
    plot_output = y_output
    y_output = np.array(y_output)
    y_output = y_output.reshape(y_output.shape[0], 1)
    print(y_output)
    train_info = train(x_input, y_output)
    print(train_info)
    predicted_value = predict([2], train_info[0])
    print(predicted_value)
    predicted_data = [i for i in range(0, 11)]
    label_predict = predicted_data 
    predicted_data = [[i] for i in predicted_data]
    predicted_data = np.array(predicted_data)
    predict_values = predict(predicted_data, train_info[0])
    plt.scatter(plot_input, plot_output)
    plt.plot(predicted_data, predict_values.flatten(), 'b')
    plt.show()
