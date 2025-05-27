import numpy as np

class ReLU:
    def __init__(self):
        self.input = None 

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, dout):
        dx = dout.copy()

        # Gradient of ReLU is 1 for positive inputs, 0 for negative inputs
        # Example:
        # self.input = np.array([[ 1.5, -0.3], [0.0, 2.2]])
        # res = (self.input > 0)
        # res -> np.array([[ True, False], [False, True]])
        dx = dx * (self.input > 0)

        return dx