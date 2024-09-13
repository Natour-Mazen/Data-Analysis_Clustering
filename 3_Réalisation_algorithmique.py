import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from sklearn.cluster import KMeans


# Q1: l’algorithme de coalescence.

def calculate_distance(point1, point2):
    """
    Cette fonction calcule la distance euclidienne entre deux points.

    Args:
        point1 (numpy.ndarray): Le premier point.
        point2 (numpy.ndarray): Le deuxième point.

    Returns:
        float: La distance euclidienne entre point1 et point2.
    """
    return np.matmul(np.transpose(point1 - point2), (point1 - point2))


def perform_coalescence(data_points, num_clusters, centroids, max_iterations=100, tolerance=1e-4):
    num_data_points, _ = data_points.shape
    updated_centroids = np.copy(centroids)
    cluster_labels = np.zeros(num_data_points, dtype=int)

    for iteration in range(max_iterations):
        old_centroids = np.copy(updated_centroids)

        # Assigner chaque point à un cluster
        for i in range(num_data_points):
            distances = [calculate_distance(data_points[i], updated_centroids[k]) for k in range(num_clusters)]
            cluster_labels[i] = np.argmin(distances)

        # Mettre à jour les centroïdes
        for k in range(num_clusters):
            points_in_cluster = data_points[cluster_labels == k]
            updated_centroids[k] = np.mean(points_in_cluster, axis=0)

        # Vérifier la convergence
        centroid_shift = np.sqrt(np.sum((updated_centroids - old_centroids) ** 2, axis=1))
        if np.all(centroid_shift <= tolerance):
            print(f"L'algorithme a convergé après {iteration} itérations.")
            break

    return cluster_labels, updated_centroids


def perform_coalescence(data_points, num_clusters, centroids, max_iterations=100, tolerance=1e-4):
    """
        Cette fonction effectue l'algorithme de coalescence sur un ensemble de points de données.

        L'algorithme de coalescence est une méthode de clustering qui vise à regrouper les points de données en clusters
        en minimisant la distance entre chaque point de données et le centroïde de son cluster. L'algorithme attribue chaque
        point de données au cluster dont le centroïde est le plus proche. Ensuite, les centroïdes sont mis à jour en fonction
        des points de données qui leur sont attribués. Ce processus est répété jusqu'à ce que les centroïdes ne changent
        plus ou que le nombre maximum d'itérations soit atteint.
        max_iterations est le nombre maximum d’itérations que l’algorithme doit effectuer avant de s’arrêter,
        tolerance est la variation minimale des centroïdes entre deux itérations consécutives pour que l’algorithme
        continue. Si la variation des centroïdes est inférieure ou égale à tolerance, l’algorithme est considéré comme
        ayant convergé et s’arrête.

        Args:
            data_points (numpy.ndarray): L'ensemble de points de données à regrouper.
            num_clusters (int): Le nombre de clusters à former.
            centroids (numpy.ndarray): Les centroïdes initiaux.

        Returns:
            tuple: Un tuple contenant deux éléments. Le premier élément est un numpy.ndarray représentant les labels de
            cluster pour chaque point de données. Le deuxième élément est un numpy.ndarray représentant les centroïdes
            mis à jour.
        """
    num_data_points, _ = data_points.shape
    updated_centroids = np.copy(centroids)
    cluster_labels = np.zeros(num_data_points, dtype=int)

    for iteration in range(max_iterations):
        old_centroids = np.copy(updated_centroids)

        # Assigner chaque point à un cluster
        for i in range(num_data_points):
            distances = [calculate_distance(data_points[i], updated_centroids[k]) for k in range(num_clusters)]
            cluster_labels[i] = np.argmin(distances)

        # Mettre à jour les centroïdes
        for k in range(num_clusters):
            points_in_cluster = data_points[cluster_labels == k]
            updated_centroids[k] = np.mean(points_in_cluster, axis=0)

        # Vérifier la convergence
        centroid_shift = np.sqrt(np.sum((updated_centroids - old_centroids) ** 2, axis=1))
        if np.all(centroid_shift <= tolerance):
            print(f"--> {iteration} itérations.")
            break

    return cluster_labels, updated_centroids


# Q2+3: application de l’algorithme de coalescence && Etude graphique du résultat de la partition.

# Géneration des données
mean_1 = np.array([2, 2])
cov_1 = np.array([[2, 0], [0, 2]])
data_cluster_1 = np.random.multivariate_normal(mean_1, cov_1, 128)

mean_2 = np.array([-4, -4])
cov_2 = np.array([[6, 0], [0, 6]])
data_cluster_2 = np.random.multivariate_normal(mean_2, cov_2, 128)

# Concate des données
data_points = np.concatenate((data_cluster_1, data_cluster_2))

# Coalescence VS Kmeans
num_clusters = 2
initial_centroids = np.array([data_points[randrange(0, 255)] for _ in range(num_clusters)])

plt.figure(figsize=(15, 5))

# Clustering de base
plt.subplot(1, 3, 1)
plt.scatter(data_points[:, 0], data_points[:, 1], marker="o")
plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.title("Clustering de base")

# Coalescence
print("Clustering par coalescence, l\'algo a convergé apres =")
cluster_labels, updated_centroids = perform_coalescence(data_points, num_clusters, initial_centroids)
plt.subplot(1, 3, 2)
plt.scatter(data_points[:, 0], data_points[:, 1], marker="o", c=cluster_labels.T, cmap='jet')
plt.scatter(updated_centroids[:, 0], updated_centroids[:, 1], marker="x", color='r', s=200, linewidths=3)
plt.title("Clustering par Coalescence")

# Kmeans
kmeans = KMeans(n_clusters=num_clusters, n_init=1, init='k-means++')
kmeans.fit(data_points)
plt.subplot(1, 3, 3)
plt.scatter(data_points[:, 0], data_points[:, 1], marker="o", c=kmeans.labels_, cmap='jet')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="x", color='r', s=200, linewidths=3)
plt.title("Clustering par Kmeans")

plt.show()

# Q4: Teste plusieurs fois l'algorithme de coalescence.
# On effectue la coalescence et on affiche les résultats
plt.figure(figsize=(10, 10))
for num_clusters in [2, 3, 4, 5]:
    initial_centroids = np.array([data_points[randrange(0, 255)] for _ in range(num_clusters)])
    print(f'Clustering par coalescence avec k = {num_clusters}, l\'algo a convergé apres =')
    cluster_labels, updated_centroids = perform_coalescence(data_points, num_clusters, initial_centroids)
    plt.subplot(2, 2, num_clusters - 1).scatter(data_points[:, 0], data_points[:, 1], marker="o", c=cluster_labels.T,
                                                cmap='jet')
    plt.subplot(2, 2, num_clusters - 1).scatter(updated_centroids[:, 0], updated_centroids[:, 1], marker="x", color='r',
                                                s=200, linewidths=3)
    plt.subplot(2, 2, num_clusters - 1).set_title(f"Clustering par Coalescence avec K = {num_clusters}")

plt.show()
