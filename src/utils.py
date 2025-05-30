from tensorflow.keras.datasets import cifar10
from PIL import Image
import os

# Charger les données CIFAR-10 (on n'utilise que le test set)
(_, _), (X_test, _) = cifar10.load_data()

# Répertoire de sortie (à adapter selon ton projet)
output_dir = "frontend/public/cifar10"
os.makedirs(output_dir, exist_ok=True)

# Nombre d'images à exporter
limit = 30

for i in range(limit):
    img_array = X_test[i]
    img = Image.fromarray(img_array)
    img.save(f"{output_dir}/{i}.png")

print(f"{limit} images CIFAR-10 ont été exportées vers {output_dir}")