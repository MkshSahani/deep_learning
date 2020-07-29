# for testing the neural network.


from NeuralNetwork import *
from utils import *
from Dense import *
from sigmoid_block import *
from MeanSquaredError import *
from activation import *
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Trainer import *
from sgd import *

import numpy as np 
import matplotlib.pyplot as plt 


def to_2d_np(a, type= "col"):
    '''
    Turns a 1D Tensor into 2D
    '''

    assert a.ndim == 1, \
    "Input tensors must be 1 dimensional"

    if type == "col":
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)


# linear_regression = NeuralNetwork(
#     layers=[Dense(neurons=1,
#                    activation=Linear())],
#     loss=MeanSquaredError(),
#     seed=20190501
# )

# nn = NeuralNetwork(
#     layers=[Dense(neurons=13,
#                    activation=Sigmoid()),
#             Dense(neurons=1,
#                    activation=Linear())],
#     loss=MeanSquaredError(),
#     seed=20190501
# )

dl = NeuralNetwork(
    layers=[Dense(neurons=13,
                   activation=SigMoid()),
            Dense(neurons=13,
                   activation=SigMoid()),
            Dense(neurons=1,
                   activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501
)

boston = load_boston()
data = boston.data
target = boston.target
features = boston.feature_names
s = StandardScaler()
data = s.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target,
                                                    test_size=0.3,
                                                    random_state=80718)

# make target 2d array
y_train, y_test = to_2d_np(y_train), to_2d_np(y_test)
# print(y_train)
# print(X_train)

trainer = Trainer(dl, SGD(dl, lr=0.001))
print("WOrking")

trainer.fit(X_train,
            y_train,
            X_test,
            y_test,
            epochs=10,
            eval_every=10,
            seed=20190501)

# print()
eval_regression_model(dl, X_test, y_test)


# trainer = Trainer(linear_regression, SGD(linear_regression, lr=0.01))

# x_data = [i for i in range(0, 30)]
# y_data = [i * 2 - 1 if i % 2 == 0 else i * 5 + 3 for i in range(0, 30)]
# x_data = [[i] for i in x_data]
# y_data = [[i] for i in y_data]


# x_data = np.array(x_data)
# y_data = np.array(y_data)

# trainer.fit(x_data,
#             y_data,
#             x_data,
#             y_data,
#             epochs=2000,
#             # eval_every=9,
#             seed=20190501)
# print()
# eval_regression_model(, x_data, y_data)
# y_data = dl.forward(x_data)
# print(y_data)


output_ = dl.forward(X_test)
print("This is input")
print(X_test)
print("This is output")
print(output_)
print("========")
# print(X_test[0])
# print(len(X_test))
# plt.scatter(X_test[0], y_test.flatten())
# plt.plot(X_test[0],output_)
# print(len(output_))
print("----------")
print(X_test[:,0])

print("----------")
print(y_test.flatten())
print("----------")
print(output_.flatten())
plt.scatter(X_test[:12, 6], y_test.flatten()[:12])
plt.plot(X_test[:12, 6], output_.flatten()[:12], 'r')
plt.xlabel("# Most Weighted Feature")
plt.ylabel("# Predicted Target.")
plt.show()