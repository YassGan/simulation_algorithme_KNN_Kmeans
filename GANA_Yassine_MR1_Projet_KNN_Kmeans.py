import pandas as pd  # Pour manipuler les données 
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Des outils nécessaires pour la phase de prétraitement
from sklearn.cluster import KMeans  # Utiliser K-Means pour faire le clustering 
from sklearn.neighbors import KNeighborsClassifier  # Utiliser K-NN pour faire la classification
from sklearn.metrics import silhouette_score  # Utiliser cette fonction qui va attribuer un score à la qualité des clusters obtenus
from sklearn.model_selection import train_test_split  # Pour faire la séparation en données de train et données de test

# Lire la dataset du Titanic qui est en CSV
titanic_df = pd.read_csv('titanic.csv')

##### Phase de prétraitement (Incluant l'élémination des données non pertinentes qui vont pas servir dans le clustering, la suppression des données nulles, l'encodage des données numériques, et la normalisation des données )
# Supprimer les colonnes non pertinentes 
titanic_df = titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Supprimer les valeurs nulles ou manquantes
titanic_df = titanic_df.dropna()

# Encoder les variables catégorielles qui ne sont pas numériques en variables numériques
label_encoder = LabelEncoder()
titanic_df['Sex'] = label_encoder.fit_transform(titanic_df['Sex'])
titanic_df['Embarked'] = label_encoder.fit_transform(titanic_df['Embarked'])

# Normaliser les données
scaler = StandardScaler()
scaled_features = scaler.fit_transform(titanic_df.drop('Survived', axis=1))

#####  // Fin de la phase de prétraitement

##### Phase de l'exécution des algorithmes (K-means et K-NN)
# Séparer les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(scaled_features, titanic_df['Survived'], test_size=0.2, random_state=42)

##### K-means clustering
def afficher_tableau_kmeans(kmeans_pred):
    # Passagers de chaque cluster
    passagers_cluster_1 = []
    passagers_cluster_2 = []
    passagers_cluster_3 = []

    # Remplir les listes des passagers de chaque cluster
    for i in range(len(kmeans_pred)):
        if kmeans_pred[i] == 0:
            passagers_cluster_1.append(f"Passager {i}")
        elif kmeans_pred[i] == 1:
            passagers_cluster_2.append(f"Passager {i}")
        else:
            passagers_cluster_3.append(f"Passager {i}")

    # Déterminer la longueur maximale des listes
    max_length = max(len(passagers_cluster_1), len(passagers_cluster_2), len(passagers_cluster_3))

    # Afficher le tableau
    print("+----------------------+----------------------+----------------------+")
    print("|      Cluster 1       |      Cluster 2       |      Cluster 3       |")
    print("+----------------------+----------------------+----------------------+")
    for i in range(max_length):
        passager_cluster_1 = passagers_cluster_1[i] if i < len(passagers_cluster_1) else ""
        passager_cluster_2 = passagers_cluster_2[i] if i < len(passagers_cluster_2) else ""
        passager_cluster_3 = passagers_cluster_3[i] if i < len(passagers_cluster_3) else ""
        print(f"|{passager_cluster_1.center(22)}|{passager_cluster_2.center(22)}|{passager_cluster_3.center(22)}|")
    print("+----------------------+----------------------+----------------------+")

# K-means avec K=2
KValue = 2
kmeans = KMeans(n_clusters=KValue, random_state=42)
kmeans.fit(X_train)

# Prédire les clusters pour les données de test
kmeans_pred = kmeans.predict(X_test)

print("Résultats K-means (K=2):")
afficher_tableau_kmeans(kmeans_pred)

# K-means avec K=3
KValue = 3
kmeans = KMeans(n_clusters=KValue, random_state=42)
kmeans.fit(X_train)

# Prédire les clusters pour les données de test
kmeans_pred = kmeans.predict(X_test)

print("Résultats K-means (K=3):")
afficher_tableau_kmeans(kmeans_pred)


##### K-NN 
# Choix de la valeur de K
NeighborsValue = 5

# KNN classification
knn = KNeighborsClassifier(n_neighbors=NeighborsValue)
knn.fit(X_train, y_train)

# Prédire les étiquettes de classe pour les données de test
knn_pred = knn.predict(X_test)

# Passagers de chaque cluster pour K-NN
passagers_survivants = [f"Passager {i}" for i, pred in enumerate(knn_pred) if pred == 1]
passagers_non_survivants = [f"Passager {i}" for i, pred in enumerate(knn_pred) if pred == 0]

# Déterminer la longueur maximale des listes
max_length = max(len(passagers_survivants), len(passagers_non_survivants))

# Afficher le tableau pour K-NN
print("\n\nRésultats K-NN:")
print("+----------------------+----------------------+")
print("|      Survivants      |    Non-Survivants    |")
print("+----------------------+----------------------+")
for i in range(max_length):
    passager_survivant = passagers_survivants[i] if i < len(passagers_survivants) else ""
    passager_non_survivant = passagers_non_survivants[i] if i < len(passagers_non_survivants) else ""
    print(f"|{passager_survivant.center(22)}|{passager_non_survivant.center(22)}|")
print("+----------------------+----------------------+")



###### Phase d'interprétation
# Évaluer la performance du clustering avec K-means (Score silhouette)
print("")
print("")
print("----- Résultats d'interprétation")

silhouette_avg = silhouette_score(X_test, kmeans_pred)
print("Silhouette Score (K-means):", silhouette_avg)

# Évaluer la performance de la classification avec KNN (Exactitude)
accuracy = knn.score(X_test, y_test)
print("Accuracy (KNN):", accuracy)
