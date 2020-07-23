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

# trainer = Trainer(linear_regression, SGD(lr=0.001))
# print("WOrking")

# trainer.fit(X_train,
#             y_train,
#             X_test,
#             y_test,
#             epochs=10,
#             eval_every=10,
#             seed=20190501)

# print()
# eval_regression_model(linear_regression, X_test, y_test)


trainer = Trainer(dl, SGD(lr=0.01))

trainer.fit(X_train,
            y_train,
            X_test,
            y_test,
            epochs=5,
            eval_every=9,
            seed=20190501)
print()
eval_regression_model(dl, X_test, y_test)