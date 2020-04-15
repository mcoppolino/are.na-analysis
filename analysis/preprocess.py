import csv
from ast import literal_eval
import random


def split_data(channels, collabs, split_proportion):
    assert(len(channels) == len(collabs))

    train_split = int(split_proportion * len(channels))

    data = list(zip(channels, collabs))
    random.shuffle(data)
    train_data = data[:train_split]
    test_data = data[train_split:]

    train_channels, train_collabs = zip(*train_data)
    test_channels, test_collabs = zip(*test_data)

    return train_channels, train_collabs, test_channels, test_collabs


def get_collaborators(collab_fp):
    with open(collab_fp, 'r') as f:
        r = csv.reader(f)
        next(r)

        channel_ids = []
        collaborators = []
        for row in r:
            channel_id = int(row[0])
            channel_ids.append(channel_id)

            collabs = literal_eval(row[1])
            collaborators.append(collabs)

    # This is the proportion of data we want to keep as training data:
    split_proportion = 0.98
    return split_data(channel_ids, collaborators, split_proportion)
