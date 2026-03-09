import numpy as np

class L1:
    def __init__(self, lam):
        self.lam = lam

    def gradient(self, W):
        return self.lam * np.sign(W)


class L2:
    def __init__(self, lam):
        self.lam = lam

    def gradient(self, W):
        return self.lam * W