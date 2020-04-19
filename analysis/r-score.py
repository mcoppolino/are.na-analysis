import os
import pickle
from argparse import ArgumentParser
import numpy as np
import math



def parse_args():
    parser = ArgumentParser()

    parser.add_argument('-input_dir', help='path to directory containing results of matrix decomposition',
                        default='../data/model')

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

#sort M_hat and T by M_hat score
def sort_by_score(mat, channel_dict, collab_dict, is_u=False, is_v=False):
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


def get_score(M_hat, T, a):
    R = 0
    for u in range(M_hat.size[0]):
        R_u = 0
        R_star = sum(1/(math.pow(2, (k/a-1))) for k in range(T[u].count(1))
        for i in range(M_hat.size[1]):
            if M_hat[u][i] >= 0.1 and T[u][i] == 1:
                R_u = R_u + (M_hat[u][i])/math.pow(2, (i)/(a-1))
        R = R + R_u/R_star

    return R

def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # load data
    data, channel_dict, collab_dict = load_data(input_dir)

    # flip dicts, both now {index: id}
    channel_dict = dict(map(reversed, channel_dict.items()))
    collab_dict = dict(map(reversed, collab_dict.items()))

    print("Sorting by score")
    M_hat = sort_by_score(data['M_hat'], channel_dict, collab_dict)

    R = get_score(M_hat, T, 1)
    print(R)

