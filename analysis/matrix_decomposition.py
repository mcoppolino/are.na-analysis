import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import os

from preprocess import get_collaborators


class SVDModel:
    def __init__(self, train_channels, train_collabs, test_channels, test_collabs, l):
        self.train_channels = train_channels
        self.train_collabs = train_collabs
        self.test_channels = test_channels
        self.test_collabs = test_collabs

        self.M, self.T = self.construct_matrix()
        self.U = None
        self.D = None
        self.V = None
        self.predictor_matrix = None

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

        # These are kind of going to act as the dimensions of the matrix:
        all_channels = self.train_channels + self.test_channels
        all_collab_lists = self.train_collabs + self.test_collabs
        all_collabs = []
        for collab_list in all_collab_lists:
            for collab in collab_list:
                if collab not in all_collabs:
                    all_collabs.append(collab)
        
        M_train = np.zeros((len(all_channels), len(all_collabs)))
        for i in range(0, len(self.train_collabs)):
            for collab in self.train_collabs[i]:
                M_train[i][all_collabs.index(collab)] = 1

        M_test = M_train.copy()
        for i in range(0, len(self.test_collabs)):
            for collab in self.test_collabs[i]:
                M_test[i + len(self.train_collabs)][all_collabs.index(collab)] = 1

        np.savetxt("M_train.csv", M_train, delimiter=",")
        np.savetxt("M_test.csv", M_test, delimiter=",")
        print("M_train matrix:")
        print(M_train)
        print("M_test matrix:")
        print(M_test)
        return M_train, M_test


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

    def calculate_predictor_matrix(self):
        self.predictor_matrix = np.matmul(self.U, self.V)
        print("predictor_matrix:")
        print(predictor_matrix)
        # Should we write the predictor_matrix to a .csv?


    def test(self):
        threshold_for_correct_prediction = 0.1
        should_be_ones = []
        total_predictions = 0
        total_correct_predictions = 0
        for i in range(0, len(self.M)):
            for j in range(0, len(self.M[i])):
                if self.T[i][j] == 1 and self.M[i][j] == 0:
                    index_tuple = (i,j)
                    prediction = self.predictor_matrix[i][j]
                    should_be_ones.append((index_tuple, prediction))

                    if prediction > 0.1:
                        total_correct_predictions += 1
                    total_predictions += 1
        
        print("Here is a list of prediction values for testing (they should be close to 1):")
        print(should_be_ones)
        print("Percentage of correct predictions: " + str(total_correct_predictions / total_predictions))

                    

def main():
    train_chan, train_collab, test_chan, test_collab = get_collaborators(os.getcwd() + "/data/csv/collaborators_with_owners.csv")
    num_components = 4000
    model = SVDModel(train_chan, train_collab, test_chan, test_collab, num_components)
    model.construct_matrix()
    model.truncated_SVD()
    model.calculate_predictor_matrix()
    model.test(model.M, model.T)





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
