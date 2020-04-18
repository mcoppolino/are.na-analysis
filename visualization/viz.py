import os
import pickle
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('-input_dir', help='path to directory containing results of matrix decomposition',
                        default='../data/model')
    parser.add_argument('-output_dir', help='path to output directory to save visuals',
                        default='./plots')

    return parser.parse_args()


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

    data_dict = np.load(npz_file, allow_pickle=True)
    [channel_dict, collab_dict] = pickle.load(open(dicts_file, 'rb'))

    return data_dict, channel_dict, collab_dict


def sort_by_ids(mat, channel_dict, collab_dict, is_u=False, is_v=False):
    collab_order = [key for (key, value) in sorted(collab_dict.items(), key=lambda x: x[1])]
    channel_order = [key for (key, value) in sorted(channel_dict.items(),  key=lambda x: x[1])]

    if not is_u:
        idx = np.empty_like(channel_order)
        idx[channel_order] = np.arange(len(channel_order))
        mat[:] = mat[:, idx]

    if not is_v:
        mat = mat.T

        idx = np.empty_like(collab_order)
        idx[collab_order] = np.arange(len(collab_order))
        mat = mat[:, idx]

        mat = mat.T

    return mat


def plot_matrix(mat, output_dir, title):
    print("Plotting %s" % title)
    if title == 'M_hat':
        plt.imshow(mat, cmap='hot', vmin=0.1)
    else:
        plt.imshow(mat, cmap='hot')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Channel')
    plt.ylabel('Collaborator')
    plt.savefig(output_dir + '/%s.png' % title)
    plt.close()


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # load data
    data, channel_dict, collab_dict = load_data(input_dir)

    # flip dicts, both now {index: id}
    channel_dict = dict(map(reversed, channel_dict.items()))
    collab_dict = dict(map(reversed, collab_dict.items()))

    print("Sorting by id")
    M = sort_by_ids(data['M'], channel_dict, collab_dict)
    M_U = sort_by_ids(data['M_U'], channel_dict, collab_dict, is_u=True)
    M_V = sort_by_ids(data['M_V'], channel_dict, collab_dict, is_v=True)
    M_hat = sort_by_ids(data['M_hat'], channel_dict, collab_dict)

    # verify out directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plot_matrix(M, output_dir, 'M')
    plot_matrix(M_U, output_dir, 'M_U')
    plot_matrix(M_V, output_dir, 'M_V')
    plot_matrix(M_hat, output_dir, 'M_hat')
    # plot_matrix(np.absolute(np.subtract(T, M_hat)), output_dir, 'Error')


if __name__ == '__main__':
    main()