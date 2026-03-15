import numpy as np  

class RMSNorm:
    def __init__(self, dim, eps=1e-8):
        self.dim = dim
        self.eps = eps
        self.gamma = np.ones((1, dim))

    def forward(self, x):
        self.x = x
        self.rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        self.normalized = x / self.rms
        return self.normalized * self.gamma

    def backward(self, dout):
        self.grad_gamma = np.sum(dout * self.normalized, axis=0, keepdims=True)

        sum_term = np.sum(dout * self.normalized, axis=-1, keepdims=True)

        dx = (self.gamma / self.rms) * dout \
             - (self.gamma * self.x / (self.dim * self.rms**3)) * sum_term

        return dx

    def update_params(self, lr):
        self.gamma -= lr * self.grad_gamma