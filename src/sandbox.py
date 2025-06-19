from tensorflow.keras.datasets import cifar10
import numpy as np
from PIL import Image
import os

# Charger CIFAR-10
(X_train, y_train), _ = cifar10.load_data()
X_train = X_train.astype('uint8')  # Pour sauvegarder en PNG

# Créer le dossier s'il n'existe pas
output_dir = "frontend/public/cifar10_V2"
os.makedirs(output_dir, exist_ok=True)

# Tirer 20 indices aléatoires
indices = np.random.choice(len(X_train), 20, replace=False)

for i, idx in enumerate(indices):
    img = X_train[idx]
    label = int(y_train[idx])
    img_pil = Image.fromarray(img)
    img_pil.save(f"{output_dir}/{i}.png")