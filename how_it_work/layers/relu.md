# ReLU – Rectified Linear Unit

La fonction **ReLU** est l'une des fonctions d'activation les plus utilisées dans les réseaux de neurones modernes, en particulier dans les **CNN**.

![alt text](images/relu.png)

## Définition

La fonction ReLU est définie par :

```
ReLU(x) = max(0, x)
```

Elle remplace toutes les valeurs négatives par zéro et laisse les positives inchangées.

## Illustration

```
Entrée : [-3, -1, 0, 2, 5]
Sortie : [ 0, 0, 0, 2, 5]
```


## Pourquoi ReLU ?

- ✅ **Simplicité** : rapide à calculer (pas d'exponentielle, pas de division)
- ✅ **Sparsité** : introduit de la non-linéarité en désactivant certains neurones (valeurs à zéro)
- ✅ **Atténue le problème du gradient vanish** : contrairement à la sigmoïde ou tanh, ReLU garde un gradient constant pour les valeurs positives

---

## Où elle intervient dans un CNN ?

Généralement **après chaque couche convolutionnelle ou pleinement connectée**, pour introduire de la non-linéarité et permettre au modèle d’apprendre des fonctions complexes.

---

## Ressources

- 📘 *Deep Learning* – Ian Goodfellow et al. 
