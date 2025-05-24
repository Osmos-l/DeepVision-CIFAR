import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# Liste des noms de classes CIFAR-10
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_and_preview_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    return x_train, y_train, x_test, y_test

def show_sample_images(x, y, num=16):
    plt.figure(figsize=(8, 8))
    for i in range(num):
        plt.subplot(4, 4, i + 1)
        plt.imshow(x[i], interpolation='nearest')
        plt.title(class_names[int(y[i])])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_class_distribution(y):
    counts = np.bincount(y.flatten())
    plt.bar(class_names, counts)
    plt.title("Distribution des classes CIFAR-10")
    plt.ylabel("Nombre d'images")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    x_train, y_train, _, _ = load_and_preview_data()
    show_sample_images(x_train, y_train)
    show_class_distribution(y_train)
