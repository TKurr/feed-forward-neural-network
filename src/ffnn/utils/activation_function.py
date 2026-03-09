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
        s = self.forward(z)
        return s * (1 - s)