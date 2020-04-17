import numpy as np
from scipy.linalg import svd


class SVDModel:
    def __init__(self, channels, collabs, trunc_len):
        print("Initializing model...")
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

    def construct_matrices_and_dicts(self):
        print('Constructing M, T, channels_dict, collab_dict...')

        # construct channel dict (guaranteed sorted, so no issues during visualization)
        num_channels = 0
        for channel_id in self.channels:
            if channel_id not in self.channel_dict:
                self.channel_dict[channel_id] = num_channels
                num_channels += 1
            else:
                print('Uh oh, channel id %i is not unique in dataset' % channel_id)

        # validate quality of channel_dict
        assert (len(self.channels) == num_channels)
        assert (len(self.channel_dict) == num_channels)

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
        self.M = np.zeros((num_collaborators + 1, num_channels + 1, ))

        # populate M
        for channel, collabs in zip(self.channels, self.collaborators):
            channel_idx = self.channel_dict[channel]
            for user in collabs:
                user_idx = self.collab_dict[user]
                self.M[user_idx][channel_idx] = 1

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
        print('Applying regular SVD...')
        self.U, self.D, self.V = svd(self.M)
        self.D = np.diag(self.D)

    def truncated_svd(self):
        self.regular_svd()
        print('Truncating U, D, V...')
        self.U = self.U[:, 0:self.trunc_len]
        self.D = self.D[0:self.trunc_len, 0:self.trunc_len]
        self.V = self.V[0:self.trunc_len, :]

    def calculate_predictions(self):
        print('Calculating predictions...')
        self.M_hat = np.matmul(self.U, self.V)
        self.M_hat /= np.linalg.norm(self.M_hat, axis=1, keepdims=True)

    def train(self):
        print("Training model...")
        self.construct_matrices_and_dicts()
        self.truncated_svd()
        self.calculate_predictions()

    def test(self, thresh):
        print("Testing model...")

        # find target test indices in M_hat (values removed from T)
        test_idxs = np.where(self.M != self.T)
        predictions = self.M_hat[test_idxs]

        num_predicted = len(test_idxs[0])
        num_correct = np.count_nonzero(predictions >= thresh)
        test_values_above_thresh = num_correct / num_predicted

        # find target non-test indices in M_hat (non test indices, zero in T, above min threshold in M_hat)
        pred_significant_thresh = 10 ** -6
        non_test_idxs = np.where((self.T == self.M) & (self.T == 0) & (self.M_hat > pred_significant_thresh))
        predictions = self.M_hat[non_test_idxs]

        num_predicted = len(non_test_idxs[0])
        num_correct = np.count_nonzero(predictions >= thresh)
        non_test_values_above_thresh = num_correct / num_predicted

        return test_values_above_thresh, non_test_values_above_thresh

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
