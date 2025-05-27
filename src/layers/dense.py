import numpy as np

class Dense:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.bias = np.zeros(output_dim)

    def forward(self, x):
        return np.dot(x, self.weights) + self.bias

    def backward(self, dout):
        # Gradients
        self.dW = np.dot(self.input.T, dout)    # [input_dim, batch] x [batch, output_dim] -> [input_dim, output_dim]
        self.db = np.sum(dout, axis=0)          # [batch, output_dim] -> [output_dim]
        dx = np.dot(dout, self.weights.T)       # [batch, output_dim] x [output_dim, input_dim] -> [batch, input_dim]

        return dx # Gradient for next layer

    def update(self, learning_rate):
        self.weights -= learning_rate * self.dW
        self.bias -= learning_rate * self.db