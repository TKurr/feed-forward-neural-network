import numpy as np

class ZeroInit:
    def initialize(self, shape):
        return np.zeros(shape)


class UniformInit:
    def __init__(self, low=-0.1, high=0.1, seed=None):
        self.low = low
        self.high = high
        self.seed = seed

    def initialize(self, shape):
        if self.seed:
            np.random.seed(self.seed)
        return np.random.uniform(self.low, self.high, shape)


class NormalInit:
    def __init__(self, mean=0.0, var=1.0, seed=None):
        self.mean = mean
        self.var = var
        self.seed = seed

    def initialize(self, shape):
        if self.seed:
            np.random.seed(self.seed)
        return np.random.normal(self.mean, np.sqrt(self.var), shape)