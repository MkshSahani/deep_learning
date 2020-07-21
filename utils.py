import numpy as np 

def assert_same_shape(array, array_grad):
    assert array.shape == array_grad.shape, \
        '''
        Two ndarrays should have the same shape;
        instead, first ndarray's shape is {0}
        and second ndarray's shape is {1}.
    '''.format(tuple(array_grad.shape), tuple(array.shape))
    return None



def permute_data(X, y):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))

def eval_regression_model(model, X_test,y_test):

    preds = model.forward(X_test)
    preds = preds.reshape(-1, 1)
    print("Mean absolute error: {:.2f}".format(mae(preds, y_test)))
    print()
    print("Root mean squared error {:.2f}".format(rmse(preds, y_test)))