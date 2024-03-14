# Compte rendu du projet "Kmeans et KNN"

**Auteur:** GANA Yassine MR1  
**Date:** 07/03/2024  

## Composition du Projet

Le projet se compose des éléments suivants :

- Un fichier contenant le dataset utilisé.
- Un fichier contenant le code d’implémentation des algorithmes K-NN et K-means.
- Ce document PDF qui explique la démarche du travail.

## Choix du Dataset

Le dataset utilisé dans ce projet est celui des passagers du Titanic, comprenant 892 items.

![Capture d'écran du dataset](https://example.com/capture_dataset.png)

## Démarche du Travail

La démarche du travail s’est déroulée comme suit :

1. Installation des bibliothèques nécessaires et l’importation des fonctionnalités qui vont être utilisées.
2. Phase de prétraitement des données, comprenant le nettoyage, l’encodage et la normalisation.
3. Exécution des algorithmes K-means et K-NN en changeant le paramètre k.

### Code d’Importation

La première étape était l’installation des bibliothèques nécessaires et l’importation des fonctionnalités qu’on va les utiliser.

![Capture du code de l’importation des fonctionnalités nécessaires](https://example.com/capture_import_code.png)

### Phase de Prétraitement

Dans la phase de prétraitement, on a effectué les tâches suivantes pour obtenir des données manipulables (nettoyage, encodage, normalisation, etc.).

![Capture de la phase de prétraitement](https://example.com/capture_preprocessing.png)

### Tournage des Algorithmes et Résultats Obtenus

#### kNN

Code d’exécution de l’algorithme kNN.

![Capture du code de l’algorithme kNN](https://example.com/capture_knn_code.png)

Résultat du kNN.

![Capture du résultat de l’algorithme kNN](https://example.com/capture_knn_result.png)

#### k-means avec k = 2

Code d’exécution de l’algorithme k-means avec k = 2.

![Capture du code de l’algorithme Kmeans avec k=2](https://example.com/capture_kmeans_k2_code.png)

Résultat du k-means avec k=2.

![Capture du résultat de l’algorithme Kmeans avec k=2](https://example.com/capture_kmeans_k2_result.png)

#### k-means avec k = 3

Code d’exécution de l’algorithme k-means avec k = 3.

![Capture du code de l’algorithme Kmeans avec k=3](https://example.com/capture_kmeans_k3_code.png)

Résultat du k-means avec k=3.

![Capture du résultat de l’algorithme Kmeans avec k=3](https://example.com/capture_kmeans_k3_result.png)

## Interprétation et Comparaison

Les résultats ont été analysés en fonction des métriques suivantes :

- Silhouette Score
- Accuracy

![Comparaison des résultats](https://example.com/comparison_results.png)

### Silhouette Score

Le Silhouette Score est utilisé pour évaluer la cohésion de la séparation des clusters obtenus par K-means. Nous avons obtenu la valeur de 0,2489.

### Accuracy

L’Accuracy est utilisée pour évaluer la performance du modèle. Cette métrique indique le pourcentage de prédictions correctes faites par le modèle par rapport au nombre total de l’échantillon.

### Comparaison des deux algorithmes

En comparant les deux approches, on peut dire que K-means fournit une structure de clustering des données, tandis que KNN cherche à prédire les classes des échantillons individuels. K-means est un algorithme de clustering non supervisé, tandis que KNN est un algorithme de classification supervisé.
