from argparse import ArgumentParser
from SVDModel import SVDModel
from preprocess import get_collaborators


def parse_args():
    parser = ArgumentParser(description='are.na data collection')

    parser.add_argument('-collabs_csv', help='path to collaborators + owners csv file to read data',
                        default='../data/csv/collaborators_with_owners.csv')
    parser.add_argument('-outdir', help='path to output directory', default='../data/model')

    parser.add_argument('-test_thresh', help='test threshold for determining predictions from M_hat',
                        default=0.1, type=float)

    return parser.parse_args()


def main():
    # parse arguments from command line
    args = parse_args()
    collabs_csv_path = args.collabs_csv
    outdir = args.outdir
    test_thresh = args.test_thresh  # TODO: find a way to optimize this

    # get data with passed in constraints (see parse_args and docstring for get_collaborators)
    channels, collabs = get_collaborators(collabs_csv_path, n=100, min_collabs=3, max_collabs=6)

    # initialize model
    model = SVDModel(channels, collabs)

    # train model (and plot singular values to find elbow point to determine optimal trunc)
    model.train()

    # test model
    test_above_thresh, non_test_above_thresh = model.test(test_thresh)

    # print test metrics
    print(test_above_thresh, non_test_above_thresh)

    # save model
    model.save(outdir)

    print('Done')


if __name__ == '__main__':
    main()
