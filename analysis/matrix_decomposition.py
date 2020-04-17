import os
from argparse import ArgumentParser
import math
import numpy as np
import pickle
from SVDModel import SVDModel
from preprocess import get_collaborators


def parse_args():
    parser = ArgumentParser(description='are.na data collection')

    parser.add_argument('-collabs_csv', help='path to collaborators + owners csv file to read data',
                        default='../data/csv/collaborators_with_owners.csv')
    parser.add_argument('-outdir', help='path to output directory', default='../data/result')

    parser.add_argument('-n', help='maximum size of data to be trained on',
                        default=None, type=int)
    parser.add_argument('-min_collabs', help='the minimum length of a collaborator list that can be trained on',
                        default=1, type=int)
    parser.add_argument('-max_collabs', help='the maximum length of a collaborator list that can be trained on',
                        default=math.inf, type=int)
    parser.add_argument('-test_thresh', help='test threshold for determining predictions from M_hat',
                        default=0.1, type=float)
    parser.add_argument('-trunc_p', help='proportion of full SVD to be used for truncated',
                        default=0.4, type=float)

    return parser.parse_args()


def main():
    # parse arguments from command line
    args = parse_args()
    collabs_csv_path = args.collabs_csv
    outdir = args.outdir
    test_thresh = args.test_thresh
    trunc_p = args.trunc_p
    n = args.n
    min_collabs = args.min_collabs
    max_collabs = args.max_collabs

    # get data with passed in constraints (see parse_args and docstring for get_collaborators)
    channels, collabs = get_collaborators(collabs_csv_path, n=n, min_collabs=min_collabs, max_collabs=max_collabs)

    # calculate the dimension of truncated SVD
    truncated_dim = int(trunc_p * len(channels))

    # initialize model
    model = SVDModel(channels, collabs, truncated_dim)

    # train model
    model.train()

    # test model
    test_above_thresh, non_test_above_thresh = model.test(test_thresh)

    # print test metrics
    print(test_above_thresh, non_test_above_thresh)

    # verify out directory exists
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # save calculated data
    print('Writing calculated data to %s' % outdir)
    npz_file = outdir + '/svd.npz'
    dicts_file = outdir + '/dicts.p'
    np.savez(npz_file, M=model.M, T=model.T, U=model.U, D=model.D, V=model.V, M_hat=model.M_hat)
    dicts = [model.channel_dict, model.collab_dict]
    pickle.dump(dicts, open(dicts_file, 'wb'))

    print('Done')


if __name__ == '__main__':
    main()
