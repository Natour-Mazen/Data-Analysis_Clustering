import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import scale

print('\n##############################################')
print('#  2.1 K-Means pour la r´eduction de couleur #')
print('##############################################\n')

# Q1 :

# On charge l'image
img = mpimg.imread('visage.bmp')

# On normalise l'image
img = np.float32(img) / 255

# On organise les données de l'image couleur sous la forme d'un tableau 256 × 256 lignes et 3 colonnes
pixels = img.reshape(-1, 3)

# Le nombre de couleurs à conserver
K = 10

# L'algorithme K-Means
kmeans = KMeans(n_clusters=K, n_init=10).fit(pixels)

# On remplace chaque couleur de l'image d'origine par son code
compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]

# On réorganiser les données de l'image couleur sous la forme d'un tableau 256 × 256 × 3
compressed_img = np.reshape(compressed_pixels, img.shape)

# Et on affiche l'image résultant de la compression
plt.figure(figsize=(6, 6))
plt.title("Compression, image Visage.bmp avec l'algorithme de K-mean, K = 10")
plt.imshow(compressed_img)
plt.show()

# Q2 :
"""
Lorsque on varie K, le nombre de couleurs dans l’image compressée change. Plus K est grand, plus de couleurs 
seront conservées dans l’image compressée, ce qui rendra l’image plus proche de l’original. Cependant, cela augmentera
également la taille du fichier de l’image compressée.
"""
# Liste des valeurs de K à tester
K_values = [2, 5, 15, 20, 50]

# Créer une nouvelle figure avec une taille spécifiée
plt.figure(figsize=(20, 20))

for i, K in enumerate(K_values):
    # l'algorithme K-Means
    kmeans = KMeans(n_clusters=K, n_init=10).fit(pixels)

    # On remplace chaque couleur de l'image d'origine par son code
    compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]

    # On réorganise les données de l'image couleur sous la forme d'un tableau 256 × 256 × 3
    compressed_img = np.reshape(compressed_pixels, img.shape)

    # Et on affiche l'image résultant de la compression
    plt.subplot(len(K_values) // 2 + len(K_values) % 2, 2, i + 1)
    plt.imshow(compressed_img)
    plt.title(f'Variation de K pour plusieurs valeur pour la compression, K = {K}')

plt.tight_layout()
plt.show()

print('\n##############################################')
print('#  2.2 Clustering de données de températures #')
print('##############################################\n')

# Chargement et prétraitement des données
data_temperature = pd.read_csv("temperatures.csv", sep=";", decimal=".", header=0, index_col=0)
data = data_temperature.drop(columns=['Region', 'Moyenne', 'Amplitude', 'Latitude', 'Longitude'])
data_scaled = scale(data)  # Normalisation des données

# Q1: On choisit la matrice de dissimilarité et construire l'arbre de classification hiérarchique
Z_complete = linkage(data_scaled, method='complete', metric='euclidean')

# Q2: On choisit le nombre de clusters et appliquer le seuil
k = 3  # Nombre de clusters
seuil_auto = Z_complete[:, 2][len(data_scaled) - k] - 1.e-10
clusters = fcluster(Z_complete, t=seuil_auto, criterion='distance')
plt.figure(figsize=(10, 7))
plt.title('Dendrogramme de la classification hiérarchique des températures avec lien Complete')
plt.xlabel('Ville')
plt.ylabel('Distance')
plt.axhline(y=seuil_auto, c='grey', lw=1, linestyle='dashed')
dendrogram(Z_complete, labels=list(data.index), color_threshold=seuil_auto)
plt.show()

# Q3: On analyse les clusters
for i in range(1, k + 1):
    print(f"Cluster {i}: {sum(clusters == i)} villes")

# Q4: On refait avec une mesure de dissimilarité différente
Z_single = linkage(data_scaled, method='ward', metric='euclidean')
seuil_auto_single = Z_single[:, 2][len(data_scaled) - k] - 1.e-10
clusters_single = fcluster(Z_single, t=seuil_auto_single, criterion='distance')
plt.figure(figsize=(10, 7))
plt.title('Dendrogramme de la classification hiérarchique des températures avec lien Ward')
plt.xlabel('Ville')
plt.ylabel('Distance')
plt.axhline(y=seuil_auto_single, c='grey', lw=1, linestyle='dashed')
dendrogram(Z_single, labels=list(data.index), color_threshold=seuil_auto_single)
plt.show()

# On trace les villes selon leurs coordonnées géographiques avec Ward
def plot_clusters(clusters, title):
    Coord = data_temperature.loc[:, ['Latitude', 'Longitude']].values
    plt.figure(figsize=(10, 7))
    plt.scatter(Coord[:, 1], Coord[:, 0], c=clusters, s=40, cmap='viridis')
    plt.title(title)
    nom_ville = list(data.index)
    for i, txt in enumerate(nom_ville):
        plt.annotate(txt, (Coord[i, 1], Coord[i, 0]))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


plot_clusters(clusters_single, 'Représentation graphique des villes selon leurs coordonnées géographiques avec Ward')
plot_clusters(clusters, 'Représentation graphique des villes selon leurs coordonnées géographiques avec Complete')

print('\n#############################')
print('#  2.3 Méthode des K-means #')
print('#############################\n')

# Q1: K-means pour le regroupement
kmeans = KMeans(n_clusters=k, n_init=10)
kmeans.fit(data_scaled)

# Q3: Affichage des partitions basées sur les coordonnées géographiques et de la température
Coord = data_temperature.loc[:, ['Latitude', 'Longitude']].values
plt.figure(figsize=(10, 7))
plt.scatter(Coord[:, 1], Coord[:, 0], c=kmeans.labels_, cmap='viridis')
plt.title('Clusters K-means en fonction des coordonnées géographiques et de la température')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Q4: Comparaison des classifications CF.Rapport pour plus de détails
for i in range(1, k + 1):
    print(f"Nombre de villes dans le cluster {i} (Hiérarchique Complete): {sum(clusters == i)}")
    print(f"Nombre de villes dans le cluster {i} (Hiérarchique Ward): {sum(clusters_single == i)}")
    print(f"Nombre de villes dans le cluster {i} (K-means): {sum(kmeans.labels_ == i - 1)}")

# Q2: La courbe d'inertie
inertias = []
K_range = range(1, 15)
for K in K_range:
    kmeans = KMeans(n_clusters=K, n_init=10)
    kmeans.fit(data_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(K_range, inertias, 'bx-')
plt.xlabel('Nombre de clusters (K)')
plt.ylabel('Inertie')
plt.title('La courbe d\'inertie pour trouver le K optimal')
plt.show()

"""
Avec la méthode de coude on trouve que la courbe se stabilise avec un nombre de cluster 
à 3
"""
