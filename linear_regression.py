
import numpy as np
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
    input_ = [i for i in range(0, 15)]
    output_ = [5 * i + 2 if i % 2 == 0 else 4 * i + 9 for i in range(0, 15)]
    input_ = [[i] for i in input_]
    output_ = [[i] for i in output_]
    input_ = np.array(input_)
    output_ = np.array(output_)
    # print(input_)
    print(output_)
    weight = train(input_, output_)
    print(weight)
    predict_ = predict(input_, weight[0])
    print(predict_)
    plt.scatter(input_, output_)
    plt.plot(input_, predict_,'g')
    plt.show()
