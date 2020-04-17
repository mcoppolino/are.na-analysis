from argparse import ArgumentParser
import math
import numpy as np
import os
from SVDModel import SVDModel
from preprocess import get_collaborators


def parse_args():
    parser = ArgumentParser(description='are.na data collection')

    parser.add_argument('-collabs_csv', help='path to collaborators + owners csv file to read data',
                        default='../data/csv/collaborators_with_owners.csv')
    parser.add_argument('-results_path', help='path to results directory to save output', default='../data/results')

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
    results_path = args.results_path
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

    # print results
    print(test_above_thresh, non_test_above_thresh)

    # initialize results directory
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # save results
    np.save(results_path + '/M', model.M)
    np.save(results_path + '/T', model.T)
    np.save(results_path + '/U', model.U)
    np.save(results_path + '/D', model.D)
    np.save(results_path + '/V', model.V)
    np.save(results_path + '/M_hat', model.M_hat)
    np.save(results_path + '/channel_dict', model.channel_dict)
    np.save(results_path + '/collab_dict', model.collab_dict)


if __name__ == '__main__':
    main()
