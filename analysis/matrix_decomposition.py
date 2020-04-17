import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import matplotlib.pyplot as plt
import pathlib
import os
import random

from preprocess import get_collaborators


class SVDModel:
    def __init__(self, channels, collabs, l):
        # self.train_channels = train_channels
        # self.train_collabs = train_collabs
        # self.test_channels = test_channels
        # self.test_collabs = test_collabs
        # self.all_channels = self.train_channels + self.test_channels
        # self.all_collabs = self.train_collabs + self.test_collabs
        self.channels = channels
        self.collabs = collabs

        self.M = None
        self.T = None
        self.U = None
        self.D = None
        self.V = None
        self.predictor_matrix = None

        self.l = l


    def pretty_print(self, matrix, title, viz=True):
        if not viz:  # print with text
            print(title)
            rows, cols = matrix.shape
            for i in range(rows):
                r = []
                for j in range(cols):
                    r.append('%.02f  ' % self.M[i][j])
                print('\t'.join(r))
            print()
        else:
            plt.figure(figsize=(3, 3))
            sns.heatmap(matrix, annot=True, fmt=".01f", cmap="Blues", cbar=False)
            plt.title(title)
            plt.show()

    def construct_matrix(self):
        # might be good to write the output of this to a csv

        # These are kind of going to act as the dimensions of the matrix:
        # all_channels = self.train_channels + self.test_channels
        # all_collab_lists = self.train_collabs + self.test_collabs
        all_collabs = []
        for collab_list in self.collabs:
            for collab in collab_list:
                if collab not in all_collabs:
                    all_collabs.append(collab)
        


        M_test = np.zeros((len(self.channels), len(all_collabs)))
        # remember that self.channels and self.collabs are of the same length, and
        # the items at each index correspond to each other
        indices_of_ones = []
        for i in range(0, len(self.channels)):
            for collab in self.collabs[i]:
                M_test[i][all_collabs.index(collab)] = 1
                indices_of_ones.append((i,all_collabs.index(collab)))


        M_train = M_test.copy()
        shuffled_indices = indices_of_ones.copy()
        random.shuffle(shuffled_indices)
        proportion_to_keep = 0.99
        small_portion_of_shuffled_indices = shuffled_indices[0:int((1 - proportion_to_keep) * len(shuffled_indices))]
        for (i, j) in small_portion_of_shuffled_indices:
            M_train[i][j] = 0



        # for i in range(0, len(self.test_collabs)):
        #     for collab in self.test_collabs[i]:
        #         M_test[i + len(self.train_collabs)][all_collabs.index(collab)] = 1

        np.savetxt("M_train.csv", M_train, delimiter=",")
        np.savetxt("M_test.csv", M_test, delimiter=",")
        print("M_train matrix:")
        print(M_train)
        print("M_test matrix:")
        print(M_test)
        self.M = M_train
        self.T = M_test


    def regular_SVD(self):
        m, n = self.M.shape
        self.U, svs, self.V = svd(M)
        self.D = np.zeros((m, n))
        for i, v in enumerate(svs):
            self.D[i, i] = v

    def truncated_SVD(self, the_matrix):
        ("Applying truncated SVD...")
        m, n = the_matrix.shape
        self.U, svs, self.V = svd(the_matrix)
        self.D = np.zeros((m, n))
        for i, v in enumerate(svs):
            self.D[i, i] = v
        self.U = self.U[:, 0:self.l]
        self.D = self.D[0:self.l, 0:self.l]
        self.V = self.V[0:self.l, :]

    def calculate_predictor_matrix(self):
        print("Calculating predictor matrix...")
        self.predictor_matrix = np.matmul(self.U, self.V)
        print("predictor_matrix:")
        print(self.predictor_matrix)
        np.savetxt("predictor_matrix.csv", self.predictor_matrix, delimiter=",")

    def test(self):
        threshold_for_correct_prediction = 0.01
        should_be_ones = []
        total_predictions = 0
        total_correct_predictions = 0
        for i in range(0, len(self.M)):
            for j in range(0, len(self.M[i])):
                # print("value in predictor matrix at this index: ", self.predictor_matrix[i][j])
                if self.T[i][j] == 1 and self.M[i][j] == 0:
                    index_tuple = (i,j)
                    prediction = self.predictor_matrix[i][j]
                    print("prediction: ", prediction)
                    should_be_ones.append((index_tuple, prediction))
                    if prediction > 0.1:
                        total_correct_predictions += 1
                    total_predictions += 1
        total_correct_random_predictions = 0
        random_non_one_indices = []
        how_many_random = 2000
        a = 0
        while a < how_many_random:
            random_i = random.randint(0,len(self.T)-1)
            random_j = random.randint(0,len(self.T[0])-1)
            if self.T[random_i][random_i] != 1:
                random_non_one_indices.append((random_i,random_j))
                a += 1
        for (i,j) in random_non_one_indices:
            if self.predictor_matrix[i][j] > threshold_for_correct_prediction:
                total_correct_random_predictions += 1
            
        
        print("Here is a list of prediction values for testing (they should be close to 1):")
        print(should_be_ones)
        print("Proportion of correct predictions: " + str(total_correct_predictions / total_predictions))
        print("Proportion of correct predictions for random indices: " + str(total_correct_random_predictions / how_many_random))
        # AT SOME POINT IT COULD BE GOOD TO CONVERT THE INDICES TO CHANNELS AND COLLABORATORS (ALBEIT UNNECESSARY)


    def generate_collaborator_recommendations(self):
        recommendation_threshold = 0.01
        indices_of_recommendations_with_recommendation_value = []
        channel_and_collaborator_with_recommendation_value = []
        all_collabs = []
        for collab_list in self.collabs:
            for collab in collab_list:
                if collab not in all_collabs:
                    all_collabs.append(collab)
        for i in range(0, len(self.predictor_matrix)):
            for j in range(0, len(self.predictor_matrix[i])):
                if self.T[i][j] != 1 and self.predictor_matrix[i][j] > recommendation_threshold:
                    indices = (i,j)
                    recommendation_value = self.predictor_matrix[i][j]
                    indices_of_recommendations.append((indices, recommendation_value))
                    channel_and_collaborator_with_recommendation_value.append(self.channels[i],all_collabs[j],recommendation_value)
        # print(sorted(indices_of_recommendations_with_recommendation_value, key=lambda a: a[1]))
        print(sorted(channel_and_collaborator_with_recommendation_value, key=lambda a: a[2]))
        return sorted(indices_of_recommendations_with_recommendation_value, key=lambda a: a[1])
        # AT SOME POINT IT WOULD BE GOOD TO CONVERT THE INDICES TO CHANNELS AND COLLABORATORS (which was done)


                    

def main():
    channels, collabs = get_collaborators(os.getcwd() + "/data/csv/collaborators_with_owners.csv")
    num_components = 4000
    model = SVDModel(channels, collabs, num_components)
    model.construct_matrix()
    model.truncated_SVD(model.M)
    model.calculate_predictor_matrix()
    model.test()
    # model.pretty_print(model.predictor_matrix, "Predictor matrix visualization")

    # This would be the code we would use to actually run our algorithm to suggest new_collaborators:
    model.truncated_SVD(model.T)
    model.calculate_predictor_matrix()
    model.generate_collaborator_recommendations()




    # # This would be the code we would use to actually run our algorithm to suggest new_collaborators:
    # num_components = 4000
    # new_model = SVDModel(channels, collabs, num_components)
    # new_model.construct_matrix()
    # new_model.truncated_SVD(new_model.T)
    # # indices_and_rec_values = new_model.generate_collaborator_recommendations()




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
