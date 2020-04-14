from ast import literal_eval
import csv
import argparse
import sys
csv.field_size_limit(sys.maxsize)

def parse_args():
    parser = argparse.ArgumentParser(description='add owners to collaborator data and save to csv')

    parser.add_argument('-collab_csv', help='path to collaborators csv file to be written to',
                        default='../data/csv/collaborators.csv')
    parser.add_argument('-channel_csv', help='path to channels csv file to be read from',
                        default='../data/csv/channels.csv')
    parser.add_argument('-out_csv', help='path to write result to',
                        default='../data/csv/collaborators_with_owners.csv')

    return parser.parse_args()


def get_channel_owners(channel_csv):
    with open(channel_csv, mode='r') as f:
        r = csv.reader(f)
        next(r)
        owners = {}
        for row in r:
            channel_id = int(row[0])
            owner = int(row[11])
            owners[channel_id] = owner

    return owners



def get_collaborators(collab_csv):
    with open(collab_csv, mode='r') as f:
        r = csv.reader(f)
        next(r)
        collab_dict = {}
        for row in r:
            channel_id = int(row[0])
            collaborators = literal_eval(row[1])
            collab_dict[channel_id] = collaborators

    return collab_dict


def write_intersect(owners, collaborators, out_csv):
    for channel_id in collaborators.keys():
        if channel_id in owners:
            collaborators[channel_id].append(owners[channel_id])

    with open(out_csv, 'w') as f:
        w = csv.writer(f, fieldnames=['channel_id', 'collaborators'])
        w.writerows(collaborators.items())


def main():
    args = parse_args()
    channel_csv = args.channel_csv
    collab_csv = args.collab_csv
    out_csv = args.out_csv

    owners = get_channel_owners(channel_csv)
    collaborators = get_collaborators(collab_csv)
    write_intersect(owners, collaborators, out_csv)


if __name__ == '__main__':
    main()