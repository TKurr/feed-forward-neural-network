import numpy as np

# Base class
class Optimizer:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update_parameters(self, weights, biases, grad_w, grad_b):
        raise NotImplementedError("Subclass must impl update_parameters")

# Vanilla GD 
# (UPDATE: theta = theta - lr(gradient_loss))
class GradientDescent(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__(lr)

    def update_parameters(self, weights, biases, grad_w, grad_b):
        
        for i in range(len(weights)):
            weights[i] -= self.lr * grad_w[i]
            biases[i] -= self.lr * grad_b[i]
            
        return weights, biases


# Adam - Adaptive Moment Estimation (BONUS)
class Adam(Optimizer):
    """
    Paper: https://arxiv.org/pdf/1412.6980

    Momentum (first moment) + variance/adaptive learning rate (second moment)
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        # default params sesuai paper
        super().__init__(lr)
        
        self.beta1 = beta1     
        self.beta2 = beta2      
        self.eps = eps         

        # State variables - diinit pas first call
        self.m_w = None   
        self.m_b = None   
        self.v_w = None   
        self.v_b = None   
        self.t = 0 # timestep

    def init_state(self, weights, biases):
        self.m_w = [np.zeros_like(w) for w in weights]
        self.m_b = [np.zeros_like(b) for b in biases]
        self.v_w = [np.zeros_like(w) for w in weights]
        self.v_b = [np.zeros_like(b) for b in biases]
        self.t = 0

    def update_parameters(self, weights, biases, grad_w, grad_b):
        
        if self.m_w is None:
            self.init_state(weights, biases)
        
        self.t += 1
        
        bias_correction_m = 1 - self.beta1 ** self.t
        bias_correction_v = 1 - self.beta2 ** self.t
        
        for i in range(len(weights)):
            # first moment update
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grad_w[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b[i]
            
            # second moment update
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grad_w[i] ** 2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grad_b[i] ** 2)
            
            # bias corrected estimates
            m_w_hat = self.m_w[i] / bias_correction_m
            m_b_hat = self.m_b[i] / bias_correction_m
            v_w_hat = self.v_w[i] / bias_correction_v
            v_b_hat = self.v_b[i] / bias_correction_v
            
            # update params
            weights[i] = weights[i] - self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.eps)
            biases[i] = biases[i] - self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)

        return weights, biases



# UNIT TEST
if __name__ == "__main__":
    # Test GD
    print("Testing GradientDescent...")
    gd = GradientDescent(lr=0.1)
    w = [np.array([[1.0, 2.0], [3.0, 4.0]])]
    b = [np.array([[0.1, 0.2]])]
    gw = [np.array([[0.5, 1.0], [0.3, 0.4]])]
    gb = [np.array([[0.05, 0.02]])]

    w_new, b_new = gd.update_parameters(w, b, gw, gb)
    assert np.allclose(w_new[0], [[0.95, 1.9], [2.97, 3.96]])
    print("GD: PASS")

    # Test Adam
    print("Testing Adam...")
    adam = Adam(lr=0.1)
    w2 = [np.array([[1.0, 2.0]])]
    b2 = [np.array([[0.1]])]
    gw2 = [np.array([[0.5, 1.0]])]
    gb2 = [np.array([[0.05]])]

    for _ in range(5):
        w2, b2 = adam.update_parameters(w2, b2, gw2, gb2)

    assert adam.t == 5  # timestep harus 5 setelah 5 call
    assert not np.allclose(w2[0], [[1.0, 2.0]])
    print("Adam: PASS")
    print("All tests passed")