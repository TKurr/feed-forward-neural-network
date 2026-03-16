import numpy as np  

class RMSNorm:
    def __init__(self, dim, eps=1e-8):
        self.dim = dim
        self.eps = eps
        self.gamma = np.ones((1, dim)) 

    def forward(self, x):
        self.x = x

        # hitung RMS  dari setiap sample
        self.rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        self.normalized = x / self.rms

        # dikali gamma biar model bisa atur skala fitur
        return self.normalized * self.gamma

    def backward(self, dout):
        m = dout.shape[0]

        self.grad_gamma = np.sum(dout * self.normalized, axis=0, keepdims=True) / m

        # gradien dari layer berikutnya
        g = dout * self.gamma
        mean_term = np.mean(g * self.normalized, axis=-1, keepdims=True)

        # hitung gradien terhadap input
        dx = (g - self.normalized * mean_term) / self.rms

        return dx

    def update_params(self, lr):
        # update parameter gamma pake gradient descent
        self.gamma -= lr * self.grad_gamma