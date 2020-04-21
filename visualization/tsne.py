from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from get_model_data import get_model_data


def run_tsne(channel_embeddings, output_dir):
    tsne = TSNE(verbose=3)
    X = tsne.fit_transform(channel_embeddings)

    kmeans = KMeans()
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='plasma')
    plt.title('t-SNE Clusters of Channel Embeddings')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    plt.savefig(output_dir + '/channel_cluster.png')


def main():
    data = get_model_data('../data/model')
    channel_embeddings = data['M_U']
    # user_embeddings = data['M_V']

    output_dir = './plots'

    # verify out directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_tsne(channel_embeddings, output_dir)


if __name__ == '__main__':
    main()
