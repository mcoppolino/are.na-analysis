import argparse
from utils import write_csv_data


COLLAB_TARGET_ATTRS = []


def parse_args():
    parser = argparse.ArgumentParser(description='are.na data_master collection')

    parser.add_argument('-csv', help='path to collaborators csv file to be written to', default='data/csv/collaborators.csv')
    parser.add_argument('-table', help='collaborator table name in database', default='collaborators')
    parser.add_argument('-db', help='path to db file to be written to', default='data/data.db')
    parser.add_argument('-batch_size', help='batch_size for requests (as large as possible for speed)', default=10000)

    return parser.parse_args()


def collaborator_request_iterator(csv_fp):
    # base_url = "http://api.are.na/v2/channels/"
    # end_url = "/collaborators"
    #
    # print('Establishing connection to channels API (collaborators)')
    #
    # with open(csv_fp, mode='w') as f:
    #     writer = csv.writer(f)
    #
    #     with open(csv_fp, mode='r') as rfp:
    #         r = csv.reader(rfp)
    #         channel_ids = [tuple(line)[0] for line in r]
    #
    #     for id in channel_ids[1:]:
    #         call = base_url + id + end_url
    #         req = requests.get(call)
    #         j_data = req.json()
    #
    #         print(j_data)
    #         users = j_data['users']
    #         for user in users:
    #             writer.writerow({'channel': id, 'collaborator': user['id']})
    pass


def write_collab_csv_to_db(csv_fp, db_fp, table_name):
    pass


def main():
    args = parse_args()

    csv_fp = args.csv
    db_fp = args.db
    table = args.table
    batch_size = args.batch_size

    channel_iterator = collaborator_request_iterator(batch_size)
    write_csv_data(csv_fp, channel_iterator, COLLAB_TARGET_ATTRS)
    write_collab_csv_to_db(csv_fp, db_fp, table)


if __name__ == '__main__':
    main()

