# Neural Network Module 

#### for using the module. 
```git 
# for using the module 
# clone this repository 
git clone https://github.com/MkshSahani/deep_learning 
```

### adding the neccessary files. 
```python

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

```

#### After importing the required files. 

```python
# define you model. 
# like here i am defining the model with 3 layers. 
# each hidden layer will have the 13 neurons. 

model = NeuralNetwork(
    layers=[Dense(neurons=13,
                   activation=SigMoid()),
            Dense(neurons=13,
                   activation=SigMoid()),
            Dense(neurons=1,
                   activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501
)

```
### Now load the data in the and train the model. 

```python 
# make a trainer class. 
train = Trainder(model, SGD(lr = 0.01))

train.fit(X_train,
            y_train,
            X_test,
            y_test,
            epochs=5,
            eval_every=9,
            seed=20190501)

# you model will be trained. 