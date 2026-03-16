import numpy as np

class MSE:
    def forward(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def backward(self, y_true, y_pred):
        return -2 * (y_true - y_pred)

class BinaryCrossEntropy:
    def forward(self, y_true, y_pred):
        eps = 1e-9
        return -np.mean(y_true*np.log(y_pred+eps) + (1-y_true)*np.log(1-y_pred+eps))

    def backward(self, y_true, y_pred):
        eps = 1e-9
        return (y_pred - y_true) / ((y_pred+eps)*(1-y_pred+eps))


class CategoricalCrossEntropy:
    def forward(self, y_true, y_pred):
        eps = 1e-9
        return -np.mean(np.sum(y_true*np.log(y_pred+eps), axis=1))

    def backward(self, y_true, y_pred):
        return y_pred - y_true