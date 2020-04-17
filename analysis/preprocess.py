import csv
from ast import literal_eval
import math


def get_collaborators(collab_fp, n=None, min_collabs=1, max_collabs=math.inf):
    """
    :param collab_fp: filepath to collaborators + owners csv
    :param n: length of data to return, from top of all data sorted by length of collaborators
    :param min_collabs: the minimum length of a collaborator list that can be returned
    :param max_collabs: the maximum length of a collaborator list that can be returned
    :return: channel_ids, collaborators of length n
    """
    if n and n < 1:
        print('get_collaborators received n < 1, but n must be at least 1. Exiting.')
        exit(0)

    if min_collabs < 1:
        print('get_collaborators received min_collabs < 1, but min_collabs must be at least 2. Exiting.')
        exit(0)

    if max_collabs < 2:
        print('get_collaborators received max_collabs < 2, but max_collabs must be at least 2. Exiting.')
        exit(0)

    if max_collabs < min_collabs:
        print('get_collaborators received max_collabs < min_collabs, but max_collabs must be at greater. Exiting.')
        exit(0)

    with open(collab_fp, 'r') as f:
        r = csv.reader(f)
        next(r)

        channel_ids = []
        collaborators = []
        for row in r:
            # convert channel id to int and append to channels list
            channel_id = int(row[0])
            channel_ids.append(channel_id)

            # evaluate string representation of collabs and append to collabs list
            collabs = literal_eval(row[1])
            collaborators.append(collabs)

    # sort channels by number of collaborators
    lengths = [len(collabs) for collabs in collaborators]
    data = list(zip(channel_ids, collaborators, lengths))
    data = sorted(data, key=lambda x: x[2], reverse=True)

    # filter channels by num collaborators
    data = [item for item in data if min_collabs <= item[2] <= max_collabs]

    if len(data) == 0:
        print('get_collaborators has nothing to return given constraints. Exiting.')
        exit(0)

    # sort data by channel id
    data = sorted(data, key=lambda x: x[0])

    # limit size of output to passed in n
    if n:
        data = data[:n]

    channel_ids, collaborators, _ = zip(*data)
    return channel_ids, collaborators

