# linear_regression.py

import numpy as np
import matplotlib.pyplot as plt


def forward_pass(input_, output_, weights):
    # find the product of the input and weight matrix.
    N = np.dot(input_, weights['W'])
    P = N + weights['B']  # add the weight with it.
    forward_info = {}  # dict. contain the data calculated in forward pass.
    forward_info['N'] = N
    forward_info['X'] = input_
    forward_info['Y'] = output_
    forward_info['P'] = P

    return forward_info


def loss_gradients(forward_info, weights):
    dLdP = -2 * (forward_info['Y'] - forward_info['P'])
    dPdB = np.ones_like(weights['B'])
    dLdB = (dLdP * dPdB).sum(axis=0)
    dPdN = np.ones_like(forward_info['N'])
    dNdW = np.transpose(forward_info['X'], (1, 0))
    dLdN = dLdP * dPdN

    dLdW = np.dot(dNdW, dLdN)
    # print(dLdW)s
    loss_grand = {}
    loss_grand['B'] = dLdB
    loss_grand['W'] = dLdW

    return loss_grand


def train(input_, output_, epoch=2000, learning_rate=0.001):
    weights = {}
    weights['W'] = np.random.rand(input_.shape[1], 1)
    weights['B'] = np.random.rand(input_.shape[1])

    for i in range(0, epoch):
        forward_info = forward_pass(input_, output_, weights)
        loss_grand = loss_gradients(forward_info, weights)

        for key in weights.keys():
            weights[key] -= learning_rate * loss_grand[key]

    return weights


def predict(input_, weights):
    return np.dot(input_, weights['W']) + weights['B']


if __name__ == '__main__':
    data = [[i] for i in range(0, 5)]
    data2 = [[i*2 - 2] if i % 2 == 0 else [i*2 + 5] for i in range(0, 5)]
    data = np.array(data)
    data2 = np.array(data2)
    weights = train(data, data2)
    print(weights)
    predict = predict(data, weights)
    plt.scatter(data, data2)
    plt.plot(data, predict)
    plt.show()
