import os
import pickle
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('-input_dir', help='path to directory containing results of matrix decomposition',
                        default='../data/result_test')
    parser.add_argument('-output_dir', help='path to output directory to save visuals',
                        default='../data/plots_test')

    return parser.parse_args()


def load_data(input_dir):
    print('Loading data from %s' % input_dir)
    npz_file = input_dir + '/svd.npz'
    with np.load(npz_file, allow_pickle=True) as data:
        M = data['M']
        T = data['T']
        U = data['U']
        D = data['D']
        V = data['V']
        M_hat = data['M_hat']

    dicts_file = input_dir + '/dicts.p'
    [channel_dict, collab_dict] = pickle.load(open(dicts_file, 'rb'))

    return M, T, U, D, V, M_hat, channel_dict, collab_dict


def sort_by_ids(mat, channel_dict, collab_dict, sort_by_channel=True, sort_by_collab=True):
    collab_order = [key for (key, value) in sorted(collab_dict.items(), key=lambda x: x[1])]
    channel_order = [key for (key, value) in sorted(channel_dict.items(),  key=lambda x: x[1])]

    if sort_by_channel:
        idx = np.empty_like(channel_order)
        idx[channel_order] = np.arange(len(channel_order))
        mat[:] = mat[:, idx]

    if sort_by_collab:
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
    M, T, U, D, V, M_hat, channel_dict, collab_dict = load_data(input_dir)

    # flip dicts, both now {index: id}
    channel_dict = dict(map(reversed, channel_dict.items()))
    collab_dict = dict(map(reversed, collab_dict.items()))

    print("Sorting by id")
    M = sort_by_ids(M, channel_dict, collab_dict)
    T = sort_by_ids(T, channel_dict, collab_dict)
    U = sort_by_ids(U, channel_dict, collab_dict, sort_by_channel=False)
    V = sort_by_ids(V, channel_dict, collab_dict, sort_by_collab=False)
    M_hat = sort_by_ids(M_hat, channel_dict, collab_dict)

    # verify out directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plot_matrix(M, output_dir, 'M')
    plot_matrix(T, output_dir, 'T')
    plot_matrix(U, output_dir, 'U')
    plot_matrix(D, output_dir, 'D')
    plot_matrix(V, output_dir, 'V')
    plot_matrix(M_hat, output_dir, 'M_hat')
    plot_matrix(np.absolute(np.subtract(T, M_hat)), output_dir, 'Error')


if __name__ == '__main__':
    main()