import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples

print('#########################################')
print('#  1.1.1 Création du corpus de test     #')
print('#########################################\n')


# Simulation de 128 individus pour chaque classe
n_individus = 128

# Première classe - Distribution normale N2((2, 2)T , 2I2)
mean_class1 = np.array([2, 2])
cov_class1 = np.array([[2, 0], [0, 2]])
class1_data = np.random.multivariate_normal(mean_class1, cov_class1, n_individus)

# Deuxième classe - Distribution normale N2((-4, -4)T , 6I2)
mean_class2 = np.array([-4, -4])
cov_class2 = np.array([[6, 0], [0, 6]])
class2_data = np.random.multivariate_normal(mean_class2, cov_class2, n_individus)

# Affichage des données
plt.scatter(class1_data[:, 0], class1_data[:, 1], marker='o', label='Classe 1')
plt.scatter(class2_data[:, 0], class2_data[:, 1], marker='x', label='Classe 2')

# Ajout d'étiquettes et légende
plt.title('Distribution des deux classes')
plt.xlabel('Mesure 1')
plt.ylabel('Mesure 2')
plt.legend()

# Affichage du graphique
plt.show()


print('\n###########################################')
print('#  1.2.1 Test de la Méthode de K-means    #')
print('###########################################\n')

# Concaténation des données des deux classes
data = np.concatenate([class1_data, class2_data])

k = 2  # Number of clusters
kmeans = KMeans(n_clusters=k, init='k-means++',n_init=10)
kmeans.fit(data)


# Pour k=2
print("Affectations des clusters :\n", kmeans.labels_)

true_labels = np.concatenate([np.zeros(n_individus), np.ones(n_individus)])

# Graphique de dispersion
plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis', marker='X')
plt.title('Résultats du clustering K-Means')
plt.xlabel('Mesure 1')
plt.ylabel('Mesure 2')
plt.show()

# Calcul du score de Rand ajusté
rand_score = adjusted_rand_score(true_labels, kmeans.labels_)
print("Score de Rand ajusté :", rand_score)


"""
n_init par défaut est de 10
Q5:
En ajustant la valeur de n_init dans KMeans, on peut observer des variations dans les résultats du clustering. 
En utilisant une approche avec plusieurs initialisations, on augmente les chances d'obtenir la meilleure 
solution en termes d'inertie, tout en minimisant le risque de rester bloqué dans un minimum local. Cependant, il est 
important de noter que cela peut entraîner une augmentation du temps de calcul. On peut expérimenter avec différentes 
valeurs de n_init, par exemple en fixant n_init=10, pour trouver la configuration optimale pour le problème de 
clustering spécifique.

On peut modifier de cette maniere
# Avec une valeur différente pour n_init
kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
kmeans.fit(data)

"""
print('\n###############################################')
print('#  1.2.2 Choix du “bon” nombre de clusters    #')
print('###############################################\n')


# En Essayant différentes valeurs de K
Sil =[]
Inertia = []
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
    kmeans.fit(data)
    cluster_labels = kmeans.labels_

    # Coefficient de silhouette
    silhouette_avg = silhouette_score(data, cluster_labels)
    Sil.append(silhouette_avg)

    # Calcul des largeurs de silhouette pour chaque observation
    sample_silhouette_values = silhouette_samples(data, cluster_labels)

    # Inertie
    inertia = kmeans.inertia_
    Inertia.append(inertia)

    print(f"Pour le nombre de clusters K = {k}, le coefficient de silhouette est {silhouette_avg}, et l'inertie est {inertia}")

    # Tracé des clusters
    plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap='viridis', edgecolors='k')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, linewidths=3, color='r')
    plt.title(f'Clustering K-Means (K={k})')
    plt.show()

# Tracé des scores de silhouette
plt.plot(range(2, 7), Sil, marker='o')
plt.title('Scores de silhouette pour différentes valeurs de K')
plt.xlabel('Nombre de clusters (K)')
plt.ylabel('Score de silhouette')
plt.show()

# Tracé des scores d'inertie
plt.plot(range(2, 7), Inertia, marker='x')
plt.title('Scores d\'inertie pour différentes valeurs de K')
plt.xlabel('Nombre de clusters (K)')
plt.ylabel('Score d\'inertie')
plt.show()


"""
Question 4
Choisir le meilleur paramètre K pour le clustering. Cela peut se faire visuellement en observant comment le coefficient
de silhouette et l'inertie changent avec le nombre de clusters. Et dans notre cas le nombre optimale de K est 4.
"""

print('\n##############################################')
print('#  1.3 Clustering ascendant hiérarchique     #')
print('##############################################\n')

# Q1 -> Q4

k = 3  # Number of clusters

Z_complete = linkage(data, method='complete', metric='euclidean')
plt.title("Clustering Ascendant Hiérarchique - Seuil Automatique")
seuil_auto = Z_complete[:, 2][256-k] - 1.e-10
dendrogram(Z_complete, color_threshold=seuil_auto)
groupes_cah_auto = fcluster(Z_complete, t=seuil_auto, criterion='distance')
plt.axhline(y=seuil_auto, c='grey', lw=1, linestyle='dashed')
plt.show()

# Q. 5: Afficher le résultat du clustering avec trois couleurs/symboles
plt.scatter(data[:, 0], data[:, 1], c=groupes_cah_auto, cmap='viridis')
plt.title("Clustering Ascendant Hiérarchique - Seuil Automatique")
plt.show()

# Q6: Comparer les résultats en K=2 clusters avec différentes méthodes de liaison
methodes = ['single', 'complete', 'average', 'ward']
plt.figure(figsize=(15, 8))
for methode in methodes:
    Z = linkage(data, method=methode, metric='euclidean')
    groupes = fcluster(Z, t=2, criterion='maxclust')
    plt.subplot(2, 2, methodes.index(methode) + 1)
    plt.title(f"Liaison {methode.capitalize()}")
    dendrogram(Z)
    plt.tight_layout()

    # Calculer le score de Rand ajusté pour la comparaison
    score = adjusted_rand_score(groupes_cah_auto, groupes)
    print(f"Score de Rand Ajusté ({methode}): {score}")

plt.show()