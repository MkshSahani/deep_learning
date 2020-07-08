# linear_regression.py
import numpy as np
import matplotlib.pyplot as plt


def init_weights(input_):
    weights = {}
    weights['W'] = np.random.rand(input_.shape[1], 1)
    weights['B'] = np.random.rand(1, input_.shape[1])

    return weights


def forward_pass(weights, input_, target_):
    N = np.dot(input_, weights['W'])
    P = N + weights['B']
    L = np.mean(np.power(target_ - P, 2))
    forward_info = {}
    forward_info['N'] = N
    forward_info['P'] = P
    forward_info['L'] = L
    forward_info['X'] = input_
    forward_info['O'] = target_

    return forward_info


def loss_gradients(forward_info, weights):

    dLdP = -2 * (forward_info['O'] - forward_info['P'])

    dPdN = np.ones_like(forward_info['N'])

    dPdB = np.ones_like(weights['B'])

    dLdN = dLdP * dPdN
    dNdW = np.transpose(forward_info['X'], (1, 0))
    dLdW = np.dot(dNdW, dLdN)

    dLdB = (dLdP*dPdB).sum(axis=0)

    loss_grad = {}
    loss_grad['W'] = dLdW
    loss_grad['B'] = dLdB

    return loss_grad


def train(input_, output_, seed=2000, learning_rate=0.001):
    weights = init_weights(input_)

    for i in range(0, seed):
        forward_info = forward_pass(weights, input_, output_)
        loss_grand = loss_gradients(forward_info, weights)
        # print(forward_info)
        # print(loss_grand)
        for key in loss_grand.keys():
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
    plt.scatter(data, data2)
    data2 = predict(data, weights)
    plt.plot(data, data2)
    plt.show()
