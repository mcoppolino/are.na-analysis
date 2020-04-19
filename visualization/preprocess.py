import os
import pickle
import numpy as np


def load_data(input_directory):
    print('Loading data from %s' % input_directory)

    npz_file = input_directory + '/svd.npz'
    dicts_file = input_directory + '/dicts.p'

    # Assert load files exist
    if not os.path.exists(npz_file):
        print('load_data failed: %s does not exist. Exiting.' % npz_file)
        exit(0)
    if not os.path.exists(dicts_file):
        print('load_data failed: %s does not exist. Exiting.' % dicts_file)
        exit(0)

    model_data = np.load(npz_file, allow_pickle=True)
    [channel_dict, collab_dict] = pickle.load(open(dicts_file, 'rb'))

    return model_data, channel_dict, collab_dict


def sort_by_ids(mat, channel_dict, collab_dict, is_u=False, is_v=False):
    collab_order = [key for (key, value) in sorted(collab_dict.items(), key=lambda x: x[1])]
    channel_order = [key for (key, value) in sorted(channel_dict.items(),  key=lambda x: x[1])]

    # don't sort U by user
    if not is_u:
        idx = np.empty_like(channel_order)
        idx[channel_order] = np.arange(len(channel_order))
        mat[:] = mat[:, idx]

    # don't sort V by channel
    if not is_v:
        mat = mat.T

        idx = np.empty_like(collab_order)
        idx[collab_order] = np.arange(len(collab_order))
        mat = mat[:, idx]

        mat = mat.T

    return mat


def get_model_data(input_dir='../data/model'):

    # load data
    npz_data, channel_dict, collab_dict = load_data(input_dir)

    # flip dicts, both now {index: id}
    channel_dict = dict(map(reversed, channel_dict.items()))
    collab_dict = dict(map(reversed, collab_dict.items()))

    # extract from NpzFile, sort data, and store in model_data
    model_data = {'M': sort_by_ids(npz_data['M'], channel_dict, collab_dict),
                  'T': sort_by_ids(npz_data['T'], channel_dict, collab_dict),
                  'M_U': sort_by_ids(npz_data['M_U'], channel_dict, collab_dict, is_u=True),
                  'T_U': sort_by_ids(npz_data['T_U'], channel_dict, collab_dict, is_u=True),
                  'M_D': npz_data['M_D'],
                  'T_D': npz_data['T_D'],
                  'M_V': sort_by_ids(npz_data['M_V'], channel_dict, collab_dict, is_v=True),
                  'T_V': sort_by_ids(npz_data['T_V'], channel_dict, collab_dict, is_v=True),
                  'M_hat': sort_by_ids(npz_data['M_hat'], channel_dict, collab_dict),
                  'T_hat': sort_by_ids(npz_data['T_hat'], channel_dict, collab_dict),
                  }

    return model_data


# def plot_matrix(mat, output_dir, title):
#     print("Plotting %s" % title)
#
#     # verify out directory exists
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     if title == 'M_hat':
#         plt.imshow(mat, cmap='hot', vmin=0.1)
#     else:
#         plt.imshow(mat, cmap='hot')
#     plt.colorbar()
#     plt.title(title)
#     plt.xlabel('Channel')
#     plt.ylabel('Collaborator')
#     plt.savefig(output_dir + '/%s.png' % title)
#     plt.close()