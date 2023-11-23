# Cluster analysis
# K-means clustering

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from step_1 import raw_data


def plot_clusters(data):
    plt.figure(figsize=(18, 12))
    for i, k in enumerate([5, 7]):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(raw_data)
        labels, centers = kmeans.labels_, kmeans.cluster_centers_
        plt.subplot(2, 1, i + 1)
        plt.scatter(range(len(data)), data[:, 0], c=labels)
        plt.scatter(range(len(centers)), centers[:, 0], c='red')
        plt.title(f"K-means clustering for k = {k}")
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    correlation_matrix = np.corrcoef(raw_data.T)
    eig_vals, eig_vecs = np.linalg.eig(correlation_matrix)
    basic_components = np.dot(raw_data, eig_vecs[:, :3])

    # A
    plot_clusters(raw_data)

    # B
    plot_clusters(basic_components)
