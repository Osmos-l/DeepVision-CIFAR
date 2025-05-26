# Convolutional Neural Network - CNN

Un CNN est un type particulier de réseau de neurones artificiels très utilisé pour traiter des données structurées en grille, comme les images.

![alt text](images/cnn.png)

## Objectif

L'objectif principal est d'extraire automatiquement des caractéristiques importantes d'une image, telles que les bords, les formes, les textures, etc.

## Principe de fonctionnement

### Convolution

C'est l'opération fondamentale du CNN. Le réseau applique des petits filtres (ou noyaux) qui glissent sur l'image afin de détecter des motifs locaux.

### Couches de convolution

Plusieurs couches convolutionnelles sont empilées pour extraire des caractéristiques de complexité croissante. Les premières couches détectent des bords ou des textures simples, tandis que les couches plus profondes capturent des formes complexes ou même des objets entiers.

### Pooling

Pour réduire la taille des représentations intermédiaires et rendre le modèle plus robuste aux petites translations, on applique souvent des opérations de pooling (comme le max pooling), qui condensent une zone en une seule valeur représentative.

### Couches pleinement connectées

En fin de réseau, des couches pleinement connectées (semblables à un MLP) sont généralement utilisées pour effectuer la classification finale à partir des caractéristiques extraites.

---

On peut schématiquement résumer un CNN par la formule suivante :  
**CNN = [Couches convolutionnelles + Pooling] + Couches pleinement connectées (ex : MLP)**

*Note : il s'agit d'une simplification qui ne couvre pas toutes les variantes et optimisations possibles.*

## Architecture fonctionnelle
```
Image RGB (3 canaux, 32×32)
│
▼
┌───────────────────────────────┐
│ Conv2D(in=3, out=8, k=3, s=1, p=1) │ <-- Extraction de motifs locaux (bords, textures)
└───────────────────────────────┘
│
▼
┌─────────────┐
│ ReLU │ <-- Activation non-linéaire
└─────────────┘
│
▼
┌───────────────────┐
│ MaxPool2D(k=2, s=2) │ <-- Réduction de la taille spatiale (32×32 → 16×16)
└───────────────────┘
│
▼
┌───────────────────────────────┐
│ Conv2D(in=8, out=16, k=3, s=1, p=1) │ <-- Extraction de motifs plus complexes
└───────────────────────────────┘
│
▼
┌─────────────┐
│ ReLU │
└─────────────┘
│
▼
┌───────────────────┐
│ MaxPool2D(k=2, s=2) │ <-- Réduction de la taille spatiale (16×16 → 8×8)
└───────────────────┘
│
▼
┌───────────────┐
│ Flatten │ <-- Transformation 3D → 1D (8×8×16 = 1024)
└───────────────┘
│
▼
┌─────────────────────────┐
│ Dense(input=1024, output=64) │ <-- Combinaison des caractéristiques en vecteur compact
└─────────────────────────┘
│
▼
┌─────────────┐
│ ReLU │
└─────────────┘
│
▼
┌────────────────────────┐
│ Dense(input=64, output=10) │ <-- Sortie finale : scores pour les 10 classes
└────────────────────────┘
│
▼
┌─────────────┐
│ Softmax │ <-- Conversion en probabilités
└─────────────┘
│
▼
Classification (10 classes)
```

## Ressources

- [Vidéo explicative sur les CNN](https://www.youtube.com/watch?v=zG_5OtgxfAg)  
- Livre **Deep Learning** de Ian Goodfellow, Yoshua Bengio et Aaron Courville  
- Livre **Quand la machine apprend** de Yann LeCun  