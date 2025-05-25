import numpy as np

class ReLU:
    def forward(self, x):
        return np.maximum(0, x)