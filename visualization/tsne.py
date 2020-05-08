from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

from new_get_model_data import get_model_data


def run_tsne(channel_embeddings, output_dir):
    tsne = TSNE(verbose=3)
    X = tsne.fit_transform(channel_embeddings)

    kmeans = KMeans()
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    print("y_kmeans:")
    i = 0
    for i in range(0, len(y_kmeans)):
        print("index:", i)
        print("kmeans index: ", y_kmeans[i])
        print("x-axis value: ", X[:, 0][i])
        print("y-axis value: ", X[:, 1][i])


    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='plasma')
    plt.title('t-SNE Clusters of Channel Embeddings')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    plt.savefig(output_dir + '/channel_cluster.png')


def main():
    data = get_model_data('../data/model')
    channel_embeddings = np.transpose(data['M_V'])
    # user_embeddings = data['M_V']

    output_dir = './plots'

    # verify out directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_tsne(channel_embeddings, output_dir)


if __name__ == '__main__':
    main()
