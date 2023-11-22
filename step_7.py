# Cluster analysis
# K-means clustering

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from step_4 import corr
from step_1 import raw_data

# Perform k-means clustering for k = 5, 7


def k_means_clustering(data, k):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    return kmeans.labels_, kmeans.cluster_centers_


# Plot the results of clustering
def plot_clustering_results(data, labels, centers):
    plt.figure(figsize=(25, 5))
    plt.scatter(range(len(data)), data, c=labels)
    plt.scatter(range(len(centers)), centers, c='red')
    plt.show()


if __name__ == "__main__":
    all_channels = raw_data[:1000, :]
    eig_vals, eig_vecs = np.linalg.eig(corr)
    basic_components = np.dot(all_channels, eig_vecs[:, :3])

    # A
    labels, centers = k_means_clustering(all_channels, 5)
    plot_clustering_results(all_channels[:, 0], labels, centers[:, 0])
    labels, centers = k_means_clustering(all_channels, 7)
    plot_clustering_results(all_channels[:, 0], labels, centers[:, 0])

    # B
    labels, centers = k_means_clustering(basic_components, 5)
    plot_clustering_results(basic_components[:, 0], labels, centers[:, 0])
    labels, centers = k_means_clustering(basic_components, 7)
    plot_clustering_results(basic_components[:, 0], labels, centers[:, 0])