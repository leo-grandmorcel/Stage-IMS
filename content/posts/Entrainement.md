---
title: "Entrainement du modèle"
date: 2023-06-19
---
Pour s'entraîner le modèle a besoin d'images avec une taille divisible par 32.
A chaque époque le modèle va prendre un nombre définit d'images (batch) et va les traiter. Plus le batch est grand plus l'entrainement est rapide, cependant plus le batch est grand plus il faut de mémoire. Pour éviter de saturer la mémoire, on peut aussi jouer sur la taille des images. Il faut donc trouver un compromis entre la taille de l'image et le nombre d'images dans le batch. Pour cela, j'ai réalisé des expériences entre redimensionner les images en 1024x1024 pixels, ou séléctionner une partie aléatoire de l'image entre 1024, 512, et 256 pixels.

Chaque modèle a été entrainé pendant 200 époques avec un taux d'apprentissage à 0.00001.
Nos indices de mesures sont la préçision et l'IoU (Intersection over Union). L'IoU est la proportion de pixels correctement prédits par rapport au nombre total de pixels. La précision est la proportion de pixels correctement prédits par rapport au nombre de pixels prédits.
Voici les résultats obtenus, les titres Resize_1024, Patched_1024, Patched_512, Patched_256 correspondent aux méthodes d'entrainement décrites ci-dessus :

![Accuracy](/images/Accuracy.png)
![IoU](/images/MIoU.png)


On peut voir que la méthode la plus efficace est de séléctionner aléatoirement une zone de 512x512 pixels.