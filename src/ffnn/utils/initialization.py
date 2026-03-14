import numpy as np

class ZeroInit:
    def initialize(self, shape) -> list:
        # Inisialisasi semua param dengan 0
        return np.zeros(shape)


class UniformInit:
    def __init__(self, low=-0.1, high=0.1, seed=None) -> None:
        # Uniform init dengan batas [low, high]
        self.low = low
        self.high = high
        self.seed = seed

    def initialize(self, shape) -> list:
        # Return random uniform dari low ke high
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.uniform(self.low, self.high, shape)


class NormalInit:
    def __init__(self, mean=0.0, var=1.0, seed=None) -> None:
        # Normal init dengan mean dan variance
        self.mean = mean
        self.var = var
        self.seed = seed

    def initialize(self, shape) -> list:
        # Return random normal dengan mean dan std= akar(var)
        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.normal(self.mean, np.sqrt(self.var), shape)


class XavierInit:
    def __init__(self, seed=None) -> None:
        # Xavier (Glorot) initialization untuk sigmoid/tanh
        self.seed = seed

    def initialize(self, shape) -> list:
        # Xavier: limit = sqrt(6 / (fanIn + fanOut))
        fanIn = shape[0]
        fanOut = shape[1] if len(shape) > 1 else 1
        limit = np.sqrt(6.0 / (fanIn + fanOut))

        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.uniform(-limit, limit, shape)


class HeInit:
    def __init__(self, seed=None) -> None:
        # He initialization untuk ReLU activation
        self.seed = seed

    def initialize(self, shape) -> list:
        # He: std = sqrt(2 / fanIn)
        fanIn = shape[0]
        std = np.sqrt(2.0 / fanIn)

        if self.seed is not None:
            np.random.seed(self.seed)
        return np.random.normal(0.0, std, shape)