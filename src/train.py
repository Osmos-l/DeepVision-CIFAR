from tensorflow.keras.datasets import cifar10
from cnn import CNN
import numpy as np

if __name__ == "__main__":
    # Load and preprocess CIFAR-10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Reduce dataset size
    #subset_size = 5000
    #X_train = X_train[:subset_size]
    #y_train = y_train[:subset_size]

    #X_test = X_test[:subset_size]
    #y_test = y_test[:subset_size]

    # Calculate mean and std on training data (pixel-wise)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1 

    np.save("mean_cifar10.npy", mean)
    np.save("std_cifar10.npy", std)

    # Normalize train and test with training stats
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = CNN()

    #model.load_model("model.bin")

    print(f'X_train.shape: {X_train.shape}')

    # Train your model
    model.train(X_train, y_train, epochs=30, batch_size=1024, learning_rate=0.025)

    test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy after training: {test_acc:.4f}")
