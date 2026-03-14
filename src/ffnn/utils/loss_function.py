import numpy as np

class MSE:
    def forward(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def backward(self, y_true, y_pred):
        # dL/dz = -2 * (y_true - y_pred) / m
        # backward sudah return per 
        return -2 * (y_true - y_pred)

class BinaryCrossEntropy:
    def forward(self, y_true, y_pred):
        eps = 1e-9
        # y_pred = np.clip(y_pred, eps, 1 - eps) # handle biar 0 ga di log, ntar malah dapet infinit
        return -np.mean(y_true*np.log(y_pred+eps) + (1-y_true)*np.log(1-y_pred+eps))

    def backward(self, y_true, y_pred):
        # dL/dz = (y_pred - y_true) / (y_pred * (1 - y_pred))
        # return dah per sample
        eps = 1e-9
        # y_pred = np.clip(y_pred, eps, 1 - eps)
        return (y_pred - y_true) / ((y_pred+eps)*(1-y_pred+eps))


class CategoricalCrossEntropy:
    def forward(self, y_true, y_pred):
        eps = 1e-9
        # y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(np.sum(y_true*np.log(y_pred+eps), axis=1))

    def backward(self, y_true, y_pred):
        # dL/dz = y_pred - y_true (untuk softmax output)
        # dah persample
        return y_pred - y_true