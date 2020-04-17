import numpy as np
from scipy.linalg import svd
from preprocess import get_collaborators


class SVDModel:
    def __init__(self, channels, collabs, trunc_len):
        print("Initializing model")
        self.channels = channels  # guaranteed sorted
        self.collaborators = collabs
        self.trunc_len = trunc_len

        self.collab_dict = {}  # map {user id: index of collaborator in M}
        self.channel_dict = {}  # map {channel id: index of channel in M}

        self.M = None  # M[i][j] indicates user index i collaborated on channel index j
        self.T = None  # copy of M with some values removed, used for testing

        self.U = None  # results of SVD
        self.D = None
        self.V = None

        self.M_hat = None  # M_hat[i][j] indicates the likelihood that user index i should collab on channel index j

    def construct_matrix(self):
        print('Constructing M and T')

        # construct channel dict (guaranteed sorted, so no issues during visualization)
        num_channels = 0
        for channel_id in self.channels:
            if channel_id not in self.channel_dict:
                self.channel_dict[channel_id] = num_channels
                num_channels += 1
            else:
                print('Uh oh, channel id %i is not unique in dataset' % channel_id)

        # validate quality of channel_dict
        assert(len(self.channels) == num_channels)
        assert(len(self.channel_dict) == num_channels)

        # construct collab_dict
        num_collaborators = 0
        for collab_list in self.collaborators:
            for user in collab_list:
                if user not in self.collab_dict:
                    self.collab_dict[user] = num_collaborators
                    num_collaborators += 1

        # remove extra from last iteration
        num_channels -= 1
        num_collaborators -= 1

        # initialize M
        self.M = np.zeros((num_channels + 1, num_collaborators + 1))

        # populate M
        for channel, collabs in zip(self.channels, self.collaborators):
            channel_idx = self.channel_dict[channel]
            for user in collabs:
                user_idx = self.collab_dict[user]
                self.M[channel_idx][user_idx] = 1

        # initialize T
        self.T = self.M.copy()

        # remove p_remove % of adjacencies
        p_remove = 0.1
        x, y = np.where(self.T == 1)
        num_ones = len(x)
        num_to_remove = int(num_ones * p_remove)
        remove = np.random.choice(num_ones, num_to_remove)
        self.T[x[remove], y[remove]] = 0

    def regular_svd(self):
        print('Applying regular SVD')
        self.U, self.D, self.V = svd(self.M)
        self.D = np.diag(self.D)

    def truncated_svd(self):
        self.regular_svd()
        print('Truncating U, D, V')
        self.U = self.U[:, 0:self.trunc_len]
        self.D = self.D[0:self.trunc_len, 0:self.trunc_len]
        self.V = self.V[0:self.trunc_len, :]

    def calculate_predictions(self):
        print('Calculating predictions')
        self.M_hat = np.matmul(self.U, self.V)

    def test(self):
        return 0, 0
        # print("Testing model")
        # threshold_for_correct_prediction = 0.1
        # should_be_ones = []
        # num_predicted = 0
        # num_correct = 0
        # for i in range(len(self.M)):
        #     for j in range(len(self.M[i])):
        #         if self.T[i][j] == 1 and self.M[i][j] == 0:
        #             index_tuple = (i, j)
        #             prediction = self.M_hat[i][j]
        #             should_be_ones.append((index_tuple, prediction))
        #             if prediction > threshold_for_correct_prediction:
        #                 num_correct += 1
        #             num_predicted += 1
        # total_correct_random_predictions = 0
        # random_non_one_indices = []
        # how_many_random = 10000
        # a = 0
        # while a < how_many_random:
        #     random_i = random.randint(0,len(self.T)-1)
        #     random_j = random.randint(0,len(self.T[0])-1)
        #     if self.T[random_i][random_i] != 1:
        #         random_non_one_indices.append((random_i,random_j))
        #         a += 1
        # for i, j in random_non_one_indices:
        #     if self.M_hat[i][j] > threshold_for_correct_prediction:
        #         total_correct_random_predictions += 1
        #
        # # print("Here is a list of prediction values for testing (they should be close to 1):")
        # # print(should_be_ones)
        # model_accuracy = num_correct / num_predicted
        # random_accuracy = total_correct_random_predictions / how_many_random
        #
        # return model_accuracy, random_accuracy

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
    channels, collabs = get_collaborators("../data/csv/collaborators_with_owners.csv", n=100)

    trunc_p = 0.4
    truncated_dim = int(trunc_p * len(channels))

    model = SVDModel(channels, collabs, truncated_dim)
    model.construct_matrix()
    model.truncated_svd()
    model.calculate_predictions()
    model_acc, random_acc = model.test()

    np.save('../data/results/M', model.M)
    np.save('../data/results/T', model.T)
    np.save('../data/results/U', model.U)
    np.save('../data/results/D', model.D)
    np.save('../data/results/V', model.V)
    np.save('../data/results/M_hat', model.M_hat)
    np.save('../data/results/channel_dict', model.channel_dict)
    np.save('../data/results/collab_dict', model.collab_dict)


if __name__ == '__main__':
    main()
