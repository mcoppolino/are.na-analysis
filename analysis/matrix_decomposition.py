import csv

import numpy as np
import pickle
from scipy.linalg import svd
import random

from preprocess import get_collaborators


class SVDModel:
    def __init__(self, channels, collabs, l):
        print("Initializing model")
        self.channels = channels
        self.collabs = collabs

        self.M = None
        self.T = None
        self.U = None
        self.D = None
        self.V = None
        self.M_hat = None

        self.l = l

    def construct_matrix(self):
        print('Constructing M and T')
        all_collabs = []
        for collab_list in self.collabs:
            for collab in collab_list:
                if collab not in all_collabs:
                    all_collabs.append(collab)

        self.T = np.zeros((len(self.channels), len(all_collabs)))

        # channel i from self.channels[i] has collaborators self.collaborators[i]
        indices_of_ones = []
        for i in range(len(self.channels)):
            for collab in self.collabs[i]:
                self.T[i][all_collabs.index(collab)] = 1
                indices_of_ones.append((i, all_collabs.index(collab)))

        self.M = self.T.copy()
        shuffled_indices = indices_of_ones.copy()
        random.shuffle(shuffled_indices)
        proportion_to_keep = 0.99
        small_portion_of_shuffled_indices = shuffled_indices[0:int((1 - proportion_to_keep) * len(shuffled_indices))]
        for (i, j) in small_portion_of_shuffled_indices:
            self.M[i][j] = 0


    def regular_SVD(self):
        print('Applying regular SVD')
        self.U, self.D, self.V = svd(self.M)
        self.D = np.diag(self.D)

    def truncated_SVD(self):
        print("Applying truncated SVD")
        m, n = self.M.shape
        self.U, self.D, self.V = svd(self.M)
        self.D = np.diag(self.D)
        self.U = self.U[:, 0:self.l]
        self.D = self.D[0:self.l, 0:self.l]
        self.V = self.V[0:self.l, :]

    def calculate_predictions(self):
        print('Calculating predictions')
        self.M_hat = np.matmul(self.U, self.V)

    def test(self):
        print("Testing model")
        threshold_for_correct_prediction = 0.1
        should_be_ones = []
        num_predicted = 0
        num_correct = 0
        for i in range(len(self.M)):
            for j in range(len(self.M[i])):
                if self.T[i][j] == 1 and self.M[i][j] == 0:
                    index_tuple = (i, j)
                    prediction = self.M_hat[i][j]
                    should_be_ones.append((index_tuple, prediction))
                    if prediction > threshold_for_correct_prediction:
                        num_correct += 1
                    num_predicted += 1
        total_correct_random_predictions = 0
        random_non_one_indices = []
        how_many_random = 10000
        a = 0
        while a < how_many_random:
            random_i = random.randint(0,len(self.T)-1)
            random_j = random.randint(0,len(self.T[0])-1)
            if self.T[random_i][random_i] != 1:
                random_non_one_indices.append((random_i,random_j))
                a += 1
        for i, j in random_non_one_indices:
            if self.M_hat[i][j] > threshold_for_correct_prediction:
                total_correct_random_predictions += 1

        # print("Here is a list of prediction values for testing (they should be close to 1):")
        # print(should_be_ones)
        model_accuracy = num_correct / num_predicted
        random_accuracy = total_correct_random_predictions / how_many_random

        return model_accuracy, random_accuracy

    # def generate_collaborator_recommendations(self):
    #     recommendation_threshold = 0.1
    #     indices_of_recommendations_with_recommendation_value = []
    #     channel_and_collaborator_with_recommendation_value = []
    #     all_collabs = []
    #     for collab_list in self.collabs:
    #         for collab in collab_list:
    #             if collab not in all_collabs:
    #                 all_collabs.append(collab)
    #     for i in range(0, len(self.predictor_matrix)):
    #         for j in range(0, len(self.predictor_matrix[i])):
    #             if self.T[i][j] != 1 and self.predictor_matrix[i][j] > recommendation_threshold:
    #                 indices = (i,j)
    #                 recommendation_value = self.predictor_matrix[i][j]
    #                 indices_of_recommendations.append((indices, recommendation_value))
    #                 channel_and_collaborator_with_recommendation_value.append(self.channels[i],all_collabs[j],recommendation_value)
    #     # print(sorted(indices_of_recommendations_with_recommendation_value, key=lambda a: a[1]))
    #     print(sorted(channel_and_collaborator_with_recommendation_value, key=lambda a: a[2]))
    #     return sorted(indices_of_recommendations_with_recommendation_value, key=lambda a: a[1])
    #     # AT SOME POINT IT WOULD BE GOOD TO CONVERT THE INDICES TO CHANNELS AND COLLABORATORS (which was done)


def main():
    channels, collabs = get_collaborators("../data/csv/collaborators_with_owners.csv", n=1000)
    truncated_dim = 200

    model = SVDModel(channels, collabs, truncated_dim)
    model.construct_matrix()
    model.truncated_SVD()
    model.calculate_predictions()
    model_acc, random_acc = model.test()
    print("Model accuracy: %d\nRandom accuracy: %d\n" % (model_acc, random_acc))

    np.save('../data/results/M', model.M)
    np.save('../data/results/T', model.T)
    np.save('../data/results/U', model.U)
    np.save('../data/results/D', model.D)
    np.save('../data/results/V', model.V)
    np.save('../data/results/M_hat', model.M_hat)


if __name__ == '__main__':
    main()
