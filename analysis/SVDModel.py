import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd


class SVDModel:
    def __init__(self, channels, collabs, load_dir=None):
        print("Initializing model...")
        self.channels = channels  # guaranteed sorted
        self.collaborators = collabs
        self.trunc_len = int(.13 * len(self.channels))  # TODO: change to elbow point of svs plot from full run

        self.collab_dict = {}  # map {user id: index of collaborator in M}
        self.channel_dict = {}  # map {channel id: index of channel in M}

        self.M = None  # M[i][j] indicates user index i collaborated on channel index j
        self.T = None  # copy of M with some values removed, used for testing

        # SVD of M
        self.M_U = None
        self.M_D = None
        self.M_V = None
        self.M_U_trunc = None
        self.M_D_trunc = None
        self.M_V_trunc = None

        # SVD of T
        self.T_U = None
        self.T_D = None
        self.T_V = None
        self.T_U_trunc = None
        self.T_D_trunc = None
        self.T_V_trunc = None

        self.M_hat = None  # reconstruction of M using truncated U, V
        self.T_hat = None  # reconstruction of T using truncated U, V

        self.svs_plot = None

        if load_dir:
            self.load_model(load_dir)
        else:
            self.construct_matrices_and_dicts()

    def construct_matrices_and_dicts(self, dropout=0.1):
        """
        :param dropout: the proportion of adjacencies to remove from T to be trained on

        Constructs and saves M, a matrix of shape(total_collabs, total_channels) where M[i][j] == 1 indicates that
        user with id collab_dict[i] collaborates on channel with id channel_dict[j].

        T is a copy of M, with a proportion of ones removed indicated by the value of dropout. These missing indices
        will be used to measure the performance of the model, as the recommendations produced by the model should have
        high confidence at these indices.
        """
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
        self.M = np.zeros((num_collaborators + 1, num_channels + 1,))

        # populate M
        for channel, collabs in zip(self.channels, self.collaborators):
            channel_idx = self.channel_dict[channel]
            for user in collabs:
                user_idx = self.collab_dict[user]
                self.M[user_idx][channel_idx] = 1

        # initialize T
        self.T = self.M.copy()

        # remove dropout % of adjacencies
        x, y = np.where(self.T == 1)
        num_ones = len(x)
        num_to_remove = int(num_ones * dropout)
        remove = np.random.choice(num_ones, num_to_remove)
        self.T[x[remove], y[remove]] = 0

    def regular_svd(self, mat):
        """
        :param mat: a 2D numpy array to be decomposed using SVD
        :return: the singular value decomposition of mat

        Performs standard SVD on mat using package scipy.linalg
        """
        print('Applying regular SVD...')
        U, svs, V = svd(mat)
        D = np.diag(svs)
        return U, D, V

    def truncate(self, U, D, V):
        """
        :param mat: a 2D numpy array to be decomposed using truncated SVD
        :return: the truncated singular value decomposition of mat

        Performs truncated SVD on mat by first performing regular SVD, and truncating the dim of the resulting U and
        V to passed in self.trunc_len
        """
        print('Truncating U, D, V...')
        trunc_U = U[:, 0:self.trunc_len]
        trunc_D = D[0:self.trunc_len, 0:self.trunc_len]
        trunc_V = V[0:self.trunc_len, :]
        return trunc_U, trunc_D, trunc_V

    def train(self):
        """
        Trains the model by constructing self.M_hat, a reconstruction of M using truncated SVD. The resulting M_hat
        contains probabilities that user i should collaborate on channel j at M_hat[i][j]. M_hat is our model's
        recommendations, given all data. Stores the truncated SVD components and M_hat.
        """
        print("Training model...")
        self.M_U, self.M_D, self.M_V = self.regular_svd(self.M)
        self.M_U_trunc, self.M_D_trunc, self.M_V_trunc = self.truncate(self.M_U, self.M_D, self.M_V)
        self.M_hat = np.matmul(self.M_U_trunc, self.M_V_trunc)

        self.plot_singular_values(self.M_D)

    def test(self, thresh):
        """
        :param thresh: Test threshold to classify probabilities as positive

        Tests the model by cloning M into T, and removing some adjacencies. T_hat is constructed using truncated
        SVD on T. T_hat should contain high probabilities at the removed values, where we know there already exists
        an adjacency.
        """
        print("Testing model...")

        self.T_U, self.T_D, self.T_V = self.regular_svd(self.T)
        self.T_U_trunc, self.T_D_trunc, self.T_V_trunc = self.truncate(self.T_U, self.T_D, self.T_V)
        self.T_hat = np.matmul(self.T_U_trunc, self.T_V_trunc)

        # find target test indices in T_hat (values removed from T)
        test_idxs = np.where(self.M != self.T)
        predictions = self.T_hat[test_idxs]

        num_predicted = len(test_idxs[0])
        num_correct = np.count_nonzero(predictions >= thresh)
        test_values_above_thresh = num_correct / num_predicted

        # find target non-test indices in T_hat (non test indices, zero in T, above min threshold in M_hat)
        pred_significant_thresh = 10 ** -6
        non_test_idxs = np.where((self.T == self.M) & (self.T == 0) & (self.T_hat > pred_significant_thresh))
        predictions = self.T_hat[non_test_idxs]

        num_predicted = len(non_test_idxs[0])
        num_correct = np.count_nonzero(predictions >= thresh)
        non_test_values_above_thresh = num_correct / num_predicted

        # There should be a large difference between these values, indicating that predictions are not just noise
        return test_values_above_thresh, non_test_values_above_thresh

    def plot_singular_values(self, D):
        """
        Plots singular values of SVD to determine the optimal truncated dimension via inspection
        """
        singular_values = np.diag(D)
        plt.plot(singular_values)
        plt.title('Singular values of M')
        plt.xlabel('Nth Largest Singular Value')
        plt.ylabel('Value')
        # plt.show()
        plt.savefig('../visualization/svs.png')  # TODO change path to dir containing visualizations

    def save(self, output_directory):
        """
        :param output_directory: the directory to save the components of the model

        Saves all components of the model to be used for visualizations or loading
        """
        # verify out directory exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        print('Saving model to %s' % output_directory)
        npz_file = output_directory + '/svd.npz'
        dicts_file = output_directory + '/dicts.p'

        # Overwrite files
        if os.path.exists(npz_file):
            os.remove(npz_file)
        if os.path.exists(dicts_file):
            os.remove(dicts_file)

        np.savez(npz_file,
                 M=self.M,
                 T=self.T,
                 M_U=self.M_U,
                 M_D=self.M_D,
                 M_V=self.M_V,
                 M_U_trunc=self.M_U_trunc,
                 M_D_trunc=self.M_D_trunc,
                 M_V_trunc=self.M_V_trunc,
                 T_U=self.T_U,
                 T_D=self.T_D,
                 T_V=self.T_V,
                 T_U_trunc=self.T_U_trunc,
                 T_D_trunc=self.T_D_trunc,
                 T_V_trunc=self.T_V_trunc,
                 M_hat=self.M_hat,
                 T_hat=self.T_hat
                 )

        dicts = [self.channel_dict, self.collab_dict]
        pickle.dump(dicts, open(dicts_file, 'wb'))

    def load_model(self, input_directory):
        """
        :param input_directory: the directory containing all data to be loaded, from a previous save

        Loads all components of the model to be used for further calculations
        """
        print('Loading data from %s' % input_directory)

        npz_file = input_directory + '/svd.npz'
        dicts_file = input_directory + '/dicts.p'

        # Assert load files exist
        if not os.path.exists(npz_file):
            print('load_model failed: %s does not exist. Exiting.' % npz_file)
            exit(0)
        if not os.path.exists(dicts_file):
            print('load_model failed: %s does not exist. Exiting.' % dicts_file)
            exit(0)

        with np.load(npz_file, allow_pickle=True) as data:
            self.M = data['M']
            self.T = data['T']
            self.M_U = data['M_U']
            self.M_D = data['M_D']
            self.M_V = data['M_V']
            self.M_U_trunc = data['M_U_trunc']
            self.M_D_trunc = data['M_D_trunc']
            self.M_V_trunc = data['M_V_trunc']
            self.T_U = data['T_U']
            self.T_D = data['T_D']
            self.T_V = data['T_V']
            self.T_U_trunc = data['T_U_trunc']
            self.T_D_trunc = data['T_D_trunc']
            self.T_V_trunc = data['T_V_trunc']
            self.M_hat = data['M_hat']
            self.T_hat = data['T_hat']

        [self.channel_dict, self.collab_dict] = pickle.load(open(dicts_file, 'rb'))

# end class
