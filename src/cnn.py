import numpy as np

from layers.conv2d import Conv2D
from layers.maxpool2d import MaxPool2D
from layers.flatten import Flatten
from layers.dense import Dense
from layers.relu import ReLU
from layers.softmax import Softmax

class CNN:
    def __init__(self):
        self.layers = [
            # Convolutional layers followed by ReLU and Max Pooling
            Conv2D(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),

            Conv2D(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),

            Flatten(),

            # Fully connected layers followed by ReLU and Softmax (MLP)
            Dense(input_dim=8*8*16, output_dim=64),
            ReLU(),
            Dense(input_dim=64, output_dim=10),
            Softmax()
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def update(self, learning_rate):
        for layer in self.layers:
            if hasattr(layer, 'update'):
                layer.update(learning_rate)