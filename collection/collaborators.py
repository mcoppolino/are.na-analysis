import argparse
import csv
import requests
import sqlite3
import sys

from utils import write_csv_data

csv.field_size_limit(sys.maxsize)

COLLAB_TARGET_ATTRS = ['id', 'collaborators']


def parse_args():
    parser = argparse.ArgumentParser(description='are.na data collection')

    parser.add_argument('-collab_csv', help='path to collaborators csv file to be written to',
                        default='../data/csv/collaborators.csv')
    parser.add_argument('-channels_csv', help='path to channels csv file to be read from',
                        default='../data/csv/channels.csv')
    parser.add_argument('-table', help='collaborator table name in database', default='collaborators')
    parser.add_argument('-db', help='path to db file to be written to', default='../data/data.db')

    return parser.parse_args()


def collaborator_request_iterator(channels_csv_fp):
    print('Establishing connection to channels API (collaborators)')

    with open(channels_csv_fp, mode='r') as f:
        r = csv.reader(f)
        next(r)
        num_processed = 0
        for row in r:
            num_processed += 1
            if num_processed % 1000 == 0:
                print('processed %i of %i channels' % (num_processed, 319898))

            channel_id = row[0]
            has_collaboration = row[6]
            channel_slug = row[7]

            if has_collaboration == 'False':
                continue

            url = 'http://api.are.na/v2/channels/%s/collaborators/?per=300' % channel_slug

            req = requests.get(url)
            if req.status_code != 200:
                continue

            collab_json = req.json()
            if not collab_json:
                continue

            collaborators = collab_json['users']
            if not collaborators or collaborators == [None]:
                continue

            if collab_json['total_pages'] > 1:
                print("SKIPPING FOR PAGINATION: %s" % channel_id)

            collab_ids = [user['id'] for user in collaborators if user]

            yield {'id': channel_id, 'collaborators': collab_ids}


def write_collab_csv_to_db(csv_fp, db_fp, table_name):
    conn = sqlite3.connect(db_fp)
    c = conn.cursor()

    c.execute('DROP TABLE IF EXISTS %s;' % table_name)
    create_table_command = 'CREATE TABLE %s (id int NOT NULL, collaborators varchar)' % table_name
    c.execute(create_table_command)

    with open(csv_fp) as f:
        r = csv.reader(f)
        channel_data = [tuple(line) for line in r]

    print('Writing %i rows from %s to %s' % (len(channel_data), csv_fp, db_fp + '/' + table_name))
    for channel in channel_data:
        insert_command = 'INSERT INTO %s VALUES (?,?)' % table_name
        c.execute(insert_command, channel)

    conn.commit()
    print('Done\n')


def main():
    args = parse_args()

    collab_csv_fp = args.collab_csv
    channels_csv_fp = args.channels_csv
    db_fp = args.db
    table = args.table

    collab_iterator = collaborator_request_iterator(channels_csv_fp)
    write_csv_data(collab_csv_fp, collab_iterator, COLLAB_TARGET_ATTRS)
    write_collab_csv_to_db(collab_csv_fp, db_fp, table)


if __name__ == '__main__':
    main()
