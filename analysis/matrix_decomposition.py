import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import matplotlib.pyplot as plt

from preprocess import get_collaborators


class SVDModel:
    def __init__(self, train_channels, train_collabs, test_channels, test_collabs, l):
        self.train_channels = train_channels
        self.train_collabs = train_collabs
        self.test_channels = test_channels
        self.test_collabs = test_collabs

        self.M = self.construct_matrix(self.train_collabs, self.train_channels)
        self.U = None
        self.D = None
        self.V = None

        self.l = l

    def pretty_print(self, title, viz=True):
        if not viz:  # print with text
            print(title)
            rows, cols = self.M.shape
            for i in range(rows):
                r = []
                for j in range(cols):
                    r.append('%.02f  ' % self.M[i][j])
                print('\t'.join(r))
            print()
        else:
            plt.figure(figsize=(3, 3))
            sns.heatmap(self.M, annot=True, fmt=".01f", cmap="Blues", cbar=False)
            plt.title(title)
            plt.show()

    def construct_matrix(self):
        # might be good to write the output of this to a csv




    def regular_SVD(self):
        m, n = self.M.shape
        self.U, svs, self.V = svd(M)
        self.D = np.zeros((m, n))
        for i, v in enumerate(svs):
            self.D[i, i] = v

    def truncated_SVD(self):
        m, n = self.M.shape
        self.U, svs, self.V = svd(self.M)
        self.D = np.zeros((m, n))
        for i, v in enumerate(svs):
            self.D[i, i] = v
        self.U = self.U[:, 0:self.l]
        self.D = self.D[0:self.l, 0:self.l]
        self.V = self.V[0:self.l, :]

    def test(self):
        pass


def main():
    train_chan, train_collab, test_chan, test_collab = get_collaborators()
    model = SVDModel(train_chan, train_collab, test_chan, test_collab)
    model.truncated_SVD()




if __name__ == '__main__':
    main()

### Xavi ###
# import matplotlib.pyplot as plt
# import numpy as np
#
# from sklearn.datasets import fetch_lfw_people
# from sklearn.decomposition import PCA
# from sklearn.decomposition import TruncatedSVD
# from sklearn.utils.extmath import randomized_svd
#
# X = np.random.random_sample((5,5))
#
# X_copy = X.copy()
#
# # Regular PCA:
# print("X:")
# print(X)
#
# pca = PCA(n_components=2)
# pca.fit(X)
# X = pca.transform(X)
# s_v = pca.singular_values_
#
# print(s_v)
# print(X)
#
#
# Truncated SVD:
# X = np.random.randint(2, size=(6, 6))
# print("X:")
# print(X)
#
# U, Sigma, VT = randomized_svd(X, n_components=5)
# predictor_matrix = np.matmul(U, VT)
# print("predictor_matrix:")
# print(predictor_matrix)
#
#
# truncated_svd = TruncatedSVD(n_components=5)
# truncated_svd.fit(X)
# reduced_version = truncated_svd.transform(X)
# print("reduced_version:")
# print(reduced_version)
#
# new_orig = truncated_svd.inverse_transform(X)
# print("new_orig:")
# print(new_orig)
