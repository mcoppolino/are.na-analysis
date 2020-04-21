from argparse import ArgumentParser
from SVDModel import SVDModel
from preprocess import get_collaborators


def parse_args():
    parser = ArgumentParser(description='are.na data collection')

    parser.add_argument('-collabs_csv', help='path to collaborators + owners csv file to read data',
                        default='../data/csv/collaborators_with_owners.csv')
    parser.add_argument('-outdir', help='path to output directory', default='../data/model')

    return parser.parse_args()


def main():
    # parse arguments from command line
    args = parse_args()
    collabs_csv_path = args.collabs_csv
    outdir = args.outdir

    # get data with passed in constraints (see parse_args and docstring for get_collaborators)
    # channels, collabs = get_collaborators(collabs_csv_path, n=100, min_collabs=3, max_collabs=6)
    channels, collabs = get_collaborators(collabs_csv_path)

    # initialize model
    model = SVDModel(channels, collabs)

    # train model (and plot singular values to find elbow point to determine optimal trunc)
    model.train()

    # save model
    model.save(outdir)

    print('Done')


if __name__ == '__main__':
    main()
