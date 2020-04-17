import csv
from ast import literal_eval
import numpy as np



def get_collaborators(collab_fp, n = 0):
    """
    :param collab_fp: filepath to collaborators + owners csv
    :param n: length of data to return, from top of all data sorted by length of collaborators
    :return: channel_ids, collaborators of length n
    """
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

    lengths = [len(collabs) for collabs in collaborators]
    data = zip(channel_ids, collaborators, lengths)
    data = sorted(data, key=lambda x: x[2], reverse=True)
    data = data[:n]
    channel_ids, collaborators, _ = zip(*data)

    return channel_ids, collaborators

    # # This is the proportion of data we want to keep as training data:
    # split_proportion = 0.98
    # return split_data(channel_ids, collaborators, split_proportion)
