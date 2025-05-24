# Convolutional Neural Network - CNN
Type particulier de réseau de neurones artificiels trés utilisé pour traiter des données structurées en grille, comme des images.

## Objectif
Extraire automatiquement des caractéristiques (features) importantes d'une image, comme les bords, formes, textures, ...

## Comment ça marche 

### Convolution
C'est l'opération clé. Le réseau applique des petits filtres qui glissent sur l'image pour détecter des motifs locaux.

### Couches de convolution
Plusieurs couches successices extraient des caractéristiques de plus en plus complexes et abstraites. Les premières couches détectent
des bords et textures simples, les couches profodents capturent des formes complexes ou objets entiers.

### Pooling
Pour réduire la taille de l'image et rendre le réseau plus robuste aux petites translation, on applique souvent une opération de pooling
qui résume une zone en une valeur représentatives.

### Couches pleinement connectées
En fin de réseau, on utilise souvent des couches classiques pour faire la classification finale à partir des caractéristiques extraites.

On dire qu'un CNN = [Couches convolutionnelles + Pooling] + Couches pleinement connectés (ex: MLP)
**Il s'agit d'une banalisation**