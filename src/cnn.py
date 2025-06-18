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

    def train(self, X_train, y_train, epochs, batch_size, learning_rate):
        num_samples = X_train.shape[0]
    
        for epoch in range(epochs):
            # Shuffle data at each epoch
            indices = np.random.permutation(num_samples)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            epoch_loss = 0
            correct = 0
            
            for start in range(0, num_samples, batch_size):
                print(f"Epoch {epoch+1}/{epochs} : Processing batch {start // batch_size + 1}/{num_samples // batch_size + 1}", end='\r')

                end = start + batch_size
                x_batch = X_train[start:end]
                y_batch = y_train[start:end]
                
                # Forward pass
                outputs = self.forward(x_batch)
                
                # Compute loss and gradient
                loss, dout = self.cross_entropy_loss(outputs, y_batch)
                epoch_loss += loss
                
                # Backward pass
                self.backward(dout)
                
                # Update weights
                self.update(learning_rate)
                
                # Calculate accuracy
                preds = np.argmax(outputs, axis=1)
                correct += np.sum(preds == y_batch)
            
            accuracy = correct / num_samples

            self.save_model("model.bin")

            print(f"Epoch {epoch+1}/{epochs} — Loss: {epoch_loss:.4f} — Accuracy: {accuracy:.4f}")

    def cross_entropy_loss(self, probs, labels):
        batch_size = labels.shape[0]
        eps = 1e-10  # pour éviter log(0)
        correct_logprobs = -np.log(probs[np.arange(batch_size), labels] + eps)
        loss = np.sum(correct_logprobs) / batch_size

        # Gradient de la loss w.r.t. softmax input
        dout = probs.copy()
        dout[np.arange(batch_size), labels] -= 1
        dout /= batch_size

        return loss, dout

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            for i, layer in enumerate(self.layers):
                if hasattr(layer, 'save_weights'):
                    layer_type = type(layer).__name__
                    f.write(f"{i}:{layer_type}\n".encode())
                    layer.save_weights(f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            for i, layer in enumerate(self.layers):
                if hasattr(layer, 'load_weights'):
                    ident = f.readline().decode().strip()
                    expected = f"{i}:{type(layer).__name__}"
                    if ident != expected:
                        raise ValueError(f"Erreur de correspondance des couches : attendu {expected}, trouvé {ident}")
                    layer.load_weights(f)
    
    def evaluate(self, X, y):
        outputs = self.forward(X)
        preds = np.argmax(outputs, axis=1)
        accuracy = np.mean(preds == y)
        return accuracy