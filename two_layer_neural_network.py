import numpy as np 
import matplotlib.pyplot as plt 

def sigmoid(x):
    return 1 / (1 + np.exp(-1.0 * x))

def init_weights(input_size, hidden_size):
    weights = {}
    weights['W1'] = np.random.randn(input_size, hidden_size)
    weights['B1'] = np.random.randn(1, hidden_size)
    weights['W2'] = np.random.randn(hidden_size, 1)
    weights['B2'] = np.random.randn(1, 1)
    return weights

def forward_loss(X,y,weights):

    M1 = np.dot(X, weights['W1'])

    N1 = M1 + weights['B1']

    O1 = sigmoid(N1)
    
    M2 = np.dot(O1, weights['W2'])

    P = M2 + weights['B2']    

    loss = np.mean(np.power(y - P, 2))

    forward_info = {}
    forward_info['X'] = X
    forward_info['M1'] = M1
    forward_info['N1'] = N1
    forward_info['O1'] = O1
    forward_info['M2'] = M2
    forward_info['P'] = P
    forward_info['y'] = y

    return forward_info, loss

def loss_gradients(forward_info ,weights):
   
    dLdP = -2*(forward_info['y'] - forward_info['P'])
    
    dPdM2 = np.ones_like(forward_info['M2'])

    dLdM2 = dLdP * dPdM2
  
    dPdB2 = np.ones_like(weights['B2'])

    dLdB2 = (dLdP * dPdB2).sum(axis=0)
    
    dM2dW2 = np.transpose(forward_info['O1'], (1, 0))
    
    dLdW2 = np.dot(dM2dW2, dLdM2)

    dM2dO1 = np.transpose(weights['W2'], (1, 0)) 

    dLdO1 = np.dot(dLdM2, dM2dO1)
    
    dO1dN1 = sigmoid(forward_info['N1']) * (1- sigmoid(forward_info['N1']))
    
    dLdN1 = dLdO1 * dO1dN1
    
    dN1dB1 = np.ones_like(weights['B1'])
    
    dN1dM1 = np.ones_like(forward_info['M1'])
    
    dLdB1 = (dLdN1 * dN1dB1).sum(axis=0)
    
    dLdM1 = dLdN1 * dN1dM1
    
    dM1dW1 = np.transpose(forward_info['X'], (1, 0)) 

    dLdW1 = np.dot(dM1dW1, dLdM1)

    loss_gradients: Dict[str, ndarray] = {}
    loss_gradients['W2'] = dLdW2
    loss_gradients['B2'] = dLdB2
    loss_gradients['W1'] = dLdW1
    loss_gradients['B1'] = dLdB1
    
    return loss_gradients

def train(X_train, y_train ,n_iter = 1000,learning_rate = 0.01,hidden_size= 13,return_losses = False, return_weights  = False,return_scores = True,seed = 1):

    if seed:
        np.random.seed(seed)
    weights = init_weights(X_train.shape[1], hidden_size=hidden_size)
    for i in range(n_iter):
        # forward info. 
        forward_info, loss = forward_loss(X_train, y_train, weights)
        loss_grads = loss_gradients(forward_info, weights)
        for key in weights.keys():
            weights[key] -= learning_rate * loss_grads[key]
    return weights 


def predict(X, weights):
    M1 = np.dot(X, weights['W1'])
    N1 = M1 + weights['B1']
    O1 = sigmoid(N1)
    M2 = np.dot(O1, weights['W2'])

    P = M2 + weights['B2']    
    # print(f"{M1} : {N1} : {O1} : {M2} : {P}")
    return P

x_train = [[3,4,5],[3,5,6],[4,6,7],[3,4,1],[4,3,2],[5,3,2]]
y_train = [[9],[9],[10],[8],[9.5],[12]]



x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train)
print(y_train)

weights = train(x_train, y_train,hidden_size=3)
print(weights)

predict = predict(x_train, weights)
print(predict)
x_train_data = [i[0] for i in x_train]
plt.title("2 layer neural network.")
plt.scatter(x_train_data, y_train.flatten())
plt.plot(x_train_data, predict.flatten(),'r')
plt.show()
