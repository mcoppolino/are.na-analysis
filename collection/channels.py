import argparse
import csv
import sqlite3
import requests
from utils import write_csv_data


# the attributes to be collected from API call GET /v2/channels
# https://dev.are.na/documentation/channels#Block43473
CHANNEL_TARGET_ATTRS = ['id', 'title', 'created_at', 'updated_at', 'published',
                        'open', 'collaboration', 'slug', 'length', 'kind', 'status', 'user_id',
                        'follower_count', 'contents', 'collaborators']


def parse_args():
    parser = argparse.ArgumentParser(description='are.na data collection')

    parser.add_argument('-csv', help='path to channel csv file to be written to', default='data/csv/channels.csv')
    parser.add_argument('-table', help='channel table name in database', default='channels')
    parser.add_argument('-db', help='path to db file to be written to', default='data/data.db')
    parser.add_argument('-batch_size', help='batch_size for requests (as large as possible for speed)', default=10000)

    return parser.parse_args()


def channel_request_iterator(batch_size):
    """
    yields a list of channel json data of length batch_size
    """

    print('Establishing connection to channels API')

    page = 1
    url = 'http://api.are.na/v2/channels'

    payload = {'page': page, 'per': batch_size}
    req = requests.get(url, params=payload)

    if req.status_code != 200 or len(req.json()['channels']) == 0:
        print('Error establishing API connection. Skipping channel write.')

    num_pages = req.json()['total_pages']

    while True:
        print('Requesting channels (page %i of %i)' % (page, num_pages))

        payload = {'page': page, 'per': batch_size}
        page += 1

        req = requests.get(url, params=payload)
        channel_data = req.json()['channels']

        if req.status_code != 200 or len(channel_data) == 0:
            break

        for channel in channel_data:
            yield channel



def write_channel_csv_to_db(csv_fp, db_fp, table_name):
    conn = sqlite3.connect(db_fp)
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS %s;' % table_name)

    create_table_command = 'CREATE TABLE %s (' % table_name + \
                           'id int NOT NULL,' + \
                           'title varchar,' + \
                           'created_at datetime,' + \
                           'updated_at datetime,' + \
                           'published boolean,' + \
                           'open boolean,' + \
                           'collaboration boolean,' + \
                           'slug varchar,' + \
                           'length int,' + \
                           'kind varchar,' + \
                           'status varchar,' + \
                           'user_id varchar,' + \
                           'follower_count int,' + \
                           'contents varchar,' + \
                           'collaborators varchar' + \
                           ');'

    c.execute(create_table_command)

    with open(csv_fp) as channels_csv:
        reader = csv.reader(channels_csv)
        channel_data = [tuple(line) for line in reader]

    print('Writing %i rows from %s to %s' % (len(channel_data), csv_fp, db_fp + '/' + table_name))
    for channel in channel_data:
        while len(channel) < len(CHANNEL_TARGET_ATTRS):
            channel += (None,)

        insert_command = 'INSERT INTO %s VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)' % table_name
        c.execute(insert_command, channel)

    conn.commit()
    print('Done\n')


def main():
    args = parse_args()

    csv_fp = args.csv
    db_fp = args.db
    table = args.table
    batch_size = args.batch_size

    channel_iterator = channel_request_iterator(batch_size)
    write_csv_data(csv_fp, channel_iterator, CHANNEL_TARGET_ATTRS)
    write_channel_csv_to_db(csv_fp, db_fp, table)


if __name__ == '__main__':
    main()
