import numpy as np

class Softmax:
    def forward(self, x):
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        self.out = exp_x / sum_exp_x  # save for backward
        return self.out

    def backward(self, dout):
        batch_size, num_classes = dout.shape
        dx = np.zeros_like(dout)

        for i in range(batch_size):
            y = self.out[i].reshape(-1, 1) 
            jacobian = np.diagflat(y) - np.dot(y, y.T) 
            dx[i] = np.dot(jacobian, dout[i])

        return dx
