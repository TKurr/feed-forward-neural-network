import numpy as np

class Linear:
    def forward(self, z):
        return z

    def backward(self, z):
        return np.ones_like(z)


class ReLU:
    def forward(self, z):
        return np.maximum(0, z)

    def backward(self, z):
        return (z > 0).astype(float)


class Sigmoid:
    def forward(self, z):
        return 1 / (1 + np.exp(-z))

    def backward(self, z):
        s = self.forward(z)
        return s * (1 - s)


class Tanh:
    def forward(self, z):
        return np.tanh(z)

    def backward(self, z):
        return 1 - np.tanh(z)**2


class Softmax:
    def forward(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def backward(self, z):
        # biasanya disederhanakan saat pakai CCE, ini blom yakin bener
        # s = self.forward(z)
        # return s * (1 - s)

        # lukas 
        # Jacobian matrix softmax: diag(s) - s @ s.T per sample
        # buat backprop dengan CCE langsung, cukup return ones
        # karena CCE backward sudah mengandung bentuk y_pred - y_true tanpa kali activation
        # sc : https://www.bragitoff.com/2021/12/efficient-implementation-of-softmax-activation-function-and-its-derivative-jacobian-in-python/
        return np.ones_like(z)


class LeakyReLU:
    def __init__(self, alpha=0.01):
        # LeakyReLU: f(x) = x jika x > 0, else alpha*x
        # alpha -->  slope kecil untuk negative values
        self.alpha = alpha

    def forward(self, z):
        return np.where(z > 0, z, self.alpha * z)

    def backward(self, z):
        # df/dz = 1 kalo  z > 0, else alpha
        return np.where(z > 0, 1.0, self.alpha)


class ELU:
    def __init__(self, alpha=1.0):
        # ELU (Exponential Linear Unit): f(x) = x kalo x > 0, else alpha*(exp(x)-1)
        # alpha --> scale parameter
        self.alpha = alpha

    def forward(self, z):
        return np.where(z > 0, z, self.alpha * (np.exp(z) - 1))

    def backward(self, z):
        # df/dz = 1 kalo z > 0, else alpha*exp(z)
        return np.where(z > 0, 1.0, self.alpha * np.exp(z))

# sc : https://www.youtube.com/watch?v=WYHCP3W_kzE (no die relu, elu)
