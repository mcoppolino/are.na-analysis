import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans

MAX_CLUSTERS = 10
cmap = cm.get_cmap('tab10', MAX_CLUSTERS)


def visualize_clusters(data, centroids, centroid_indices):
    """
    Visualizes the data points and the calculated k-means cluster centers.
    Points with the same color are considered to be in the same cluster.
    Providing centroid locations and centroid indices will color the
    data points to match their respective cluster and plot the given centroids.

    :param data: 2D numpy array
    :param centroids: 2D numpy array of centroid locations
    :param centroid_indices: 1D numpy array of centroid indices for each data point in data
    """
    def plot_data(fig, color_map=None):
        x, y, z = np.hsplit(data, 3)
        fig.scatter(x, y, z, c=color_map)

    def plot_clusters(fig):
        x, y, z = np.hsplit(centroids, 3)
        fig.scatter(x, y, z, c="black", marker="x", alpha=1, s=200)

    plt.clf()
    cluster_plot = centroids is not None and centroid_indices is not None

    ax = plt.figure().add_subplot(111, projection='3d')
    colors_s = None

    if cluster_plot:
        if max(centroid_indices) + 1 > MAX_CLUSTERS:
            print(f"Error: Too many clusters. Please limit to fewer than {MAX_CLUSTERS}.")
            exit(1)
        colors_s = [cmap(l / MAX_CLUSTERS) for l in centroid_indices]
        plot_clusters(ax)

    plot_data(ax, colors_s)

    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_zlabel("Principal component 3")

    plot_name = "/data_clusters_sklearn"

    # Helps visualize clusters
    plt.gca().invert_xaxis()
    plt.savefig(output_dir + plot_name + ".png")
    plt.show()


def elbow_point_plot(cluster, errors):
    """
    This function helps create a plot representing the tradeoff between the
    number of clusters and the inertia values.

    :param cluster: 1D np array that represents K (the number of clusters)
    :param errors: 1D np array that represents the inertia values
    """
    plt.clf()
    plt.plot(cluster, errors)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.savefig(output_dir + "/elbow_plot.png")
    plt.show()


def inertia(centroids, centroid_indices):
    """
    Returns the inertia of the clustering. Inertia is defined as the
    sum of the squared distances between each data point and the centroid of
    its assigned cluster.

    :param centroids - the coordinates that represent the center of the clusters
    :param centroid_indices - the index of the centroid that corresponding data point it closest to
    :return inertia as a float
    """
    inertia_sum = 0
    for i in range(len(centroid_indices)):
        inertia_sum += (self.euclidean_dist(self.X[i], centroids[centroid_indices[i]])) ** 2
    return inertia_sum


def standardize(X):
    x_min = np.amin(X, axis=0)
    x_max = np.amax(X, axis=0)

    new_data = (data - x_min) / (x_max - x_min)
    return new_data


def cluster(X):
    """
    Performs k-means clustering on the 2D numpy array X. It produces
    elbow_point_plot used to determine the optimal number of clusters K,
    performs sk_learn_cluster using X and K, and creates a scatter
    plot with clusters.

    :param X: 2D numpy array after dimensionality reduction
    """
    X = standardize(X)
    # Use elbow_point_plot() to determine the optimal K (number of clusters)
    inertias = []
    for i in range(1, MAX_CLUSTERS + 1):
        centroids, idx = np.asarray(sk_learn_cluster(X, i))
        inertias.append(inertia(centroids, idx))

    elbow_point_plot(np.arange(1, MAX_CLUSTERS + 1), inertias)

    K = ## optimal number of clusters

    centroids, idx = np.asarray(sk_learn_cluster(X, K))
    visualize_clusters(X, centroids, idx)


def main():
    ##### I think this is how it works? (https://www.kaggle.com/leogal/pca-svd-intro-and-visualization)
    ##### Given original data A to be dim-reduced:
    ##### U, S, VT = np.linalg.svd(A)
    ##### X = np.dot(np.transpose(U), A)

    cluster(X)
    print('Number of principal components', len(S))
    print('First principal component explains', np.sum(S[0]) / np.sum(S), 'of the total variance.')
    print('Second principal component explains', np.sum(S[1]) / np.sum(S), 'of the total variance.')
    print('Third principal component explains', np.sum(S[2]) / np.sum(S), 'of the total variance.')


if __name__ == '__main__':
    main()
