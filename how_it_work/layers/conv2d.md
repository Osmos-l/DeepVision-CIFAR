# Conv2D

La couche **Conv2D** applique une **convolution sur deux dimensions spatiales** (hauteur × largeur), ce qui est parfaitement adapté au traitement des images.

## Fonctionnement

Un filtre (ou noyau) glisse sur l’image d’entrée pour détecter des motifs locaux (bords, textures, formes, etc.). Plusieurs filtres permettent d’extraire différentes caractéristiques visuelles.

## Hyperparamètres

- **in_channels** : nombre de canaux en entrée  
  Exemple : 3 pour une image RGB (Rouge, Vert, Bleu)

- **out_channels** : nombre de filtres, c’est-à-dire le nombre de motifs à détecter  
  👉 Souvent choisi comme une **puissance de 2** (ex : 8, 16, 32, 64…) pour optimiser les performances

- **kernel_size** : taille du filtre (ex : 3×3 ou 5×5)  
  Définit la zone locale analysée par le filtre

- **stride** : pas de déplacement du filtre  
  Un stride de 1 → le filtre se déplace pixel par pixel  
  Un stride de 2 → le filtre saute 1 pixel à chaque déplacement

- **padding** : ajout de zéros autour de l’image pour contrôler la taille de sortie  
  Permet d’éviter que les bords de l’image soient "mangés" par la convolution  

---

💡 *Sans padding, une convolution réduit les dimensions, surtout en bordure. Le padding permet de préserver l’information périphérique de l’image.*

## Poids
out_channels = 8, in_channels = 3, kernel_size = 3
```
┌────────────────────────────┐
│ Filtre 1                   │
│ ┌─────────┐               │
│ │ 3×3 Matrice pour canal 1│
│ │ 3×3 Matrice pour canal 2│
│ │ 3×3 Matrice pour canal 3│
│ └─────────┘               │
├────────────────────────────┤
│ Filtre 2                   │
│ ┌─────────┐               │
│ │ 3×3 Matrice pour canal 1│
│ │ 3×3 Matrice pour canal 2│
│ │ 3×3 Matrice pour canal 3│
│ └─────────┘               │
├────────────────────────────┤
│ ... jusqu'à filtre 8       │
└────────────────────────────┘
```