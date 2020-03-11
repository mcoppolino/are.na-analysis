import argparse
import csv
import os
import requests
import sqlite3
import sys

csv.field_size_limit(sys.maxsize)

# the attributes to be collected from API call GET /v2/channels
# https://dev.are.na/documentation/channels#Block43473
CHANNEL_TARGET_ATTRS = ['id', 'title', 'created_at', 'updated_at', 'published',
    'open', 'collaboration', 'slug', 'length', 'kind', 'status', 'user_id', \
    'follower_count', 'contents', 'collaborators']

# the user attributes to be collected from API call GET /v2/search (returns users)
# https://dev.are.na/documentation/search#Block59799
USER_TARGET_ATTRS = ['id', 'slug', 'username', 'first_name', 'last_name', \
    'channel_count', 'following_count', 'profile_id', 'follower_count']

BLOCK_CONN_TARGET_ATTRS = ['block_connections']

CHANNEL_CONN_TARGET_ATTRS = ['channel_connections']


def parse_args():
    parser = argparse.ArgumentParser(description='are.na data collection')
    parser.add_argument('-channel_csv', help='path to channel csv file to be written to', default='csv/channels.csv')
    parser.add_argument('-user_csv', help='path to user csv file to be written to', default='csv/users.csv')
    parser.add_argument('-block_conn_csv', help='path to block connections csv file to be written to', default='csv/block_connections.csv')
    parser.add_argument('-channel_conn_csv', help='path to channel connections csv file to be written to', default='csv/channel_connections.csv')

    parser.add_argument('-channel_table', help='channel table name in database', default='channels')
    parser.add_argument('-user_table', help='user table name in database', default='users')
    parser.add_argument('-connections_table', help='connections table name in database', default='connections')

    parser.add_argument('-db', help='path to db file to be written to', default='data.db')
    parser.add_argument('-batch_size', help='batch_size for requests (as large as possible for speed)', default=10000)

    return parser.parse_args()


def channel_request_iterator(batch_size):
    """
    yields a list of channel json data of length batch_size
    """

    print('Establishing connection to channels API')

    page = 1
    url = 'http://api.are.na/v2/channels'

    payload = {'page':page, 'per':batch_size}
    req = requests.get(url, params=payload)

    if req.status_code != 200 or len(req.json()['channels']) == 0:
        print('Error establishing API connection. Skipping channel write.')

    num_pages = req.json()['total_pages']

    while True:
        print('Requesting channels (page %i of %i)' % (page, num_pages))

        payload = {'page':page, 'per':batch_size}
        page += 1

        req = requests.get(url, params=payload)
        channel_data = req.json()['channels']

        if req.status_code != 200 or len(channel_data) == 0:
            break

        print('Writing channel data to csv')

        for channel in channel_data:
            yield channel


def user_request_iterator(batch_size):
    """
    yields a list of user json data of length batch_size
    """

    print('Establishing connection to search API (to collect users)')

    for letter in 'abcdefghijklmnopqrstuvwxyz0123456789':
        page = 1
        print('Fetching users with query "%s"' % letter)
        while True:
            url = 'http://api.are.na/v2/search/users/'
            payload = {'q':letter, 'page':page, 'per':batch_size}


            req = requests.get(url, params=payload)

            user_json = req.json()
            user_data = user_json['users']
            num_pages = user_json['total_pages']

            if req.status_code != 200 or len(user_data) == 0:
                break

            print('Writing user data to csv (page %i of %i)' % (page, num_pages))
            page += 1

            for user in user_data:
                yield user


def block_connections_iterator(channels_csv_fp, batch_size):

    print("Requesting block connections from API")

    with open(channels_csv_fp, mode='r') as f:
        reader = csv.reader(f)
        next(reader)

        channel_ids = [int(tuple(line)[0]) for line in reader]

    for id in channel_ids[1:]:
        page = 1
        out = [id]

        print('Requesting block connections for channel id %i' % id)

        while True:
            url = 'http://api.are.na/v2/channels/%s/channels' % id
            payload = {'page':page, 'per':batch_size}
            page += 1

            req = requests.get(url, params=payload)
            block_connections = req.json()['channels']

            if req.status_code != 200 or len(block_connections) == 0:
                break

            out.extend([item['channel']['id'] for item in block_connections])

        if len(out) > 1:
            print('Writing block connections to csv')
            yield {'id':id, 'block_connections':out}

def channel_connections_iterator(channels_csv_fp, batch_size):
    with open(channels_csv_fp, mode='r') as f:
        reader = csv.reader(f)
        next(reader)

        channel_ids = [int(tuple(line)[0]) for line in reader]

    for id in channel_ids[1:]:
        page = 1
        out = [id]

        print('Requesting channel connections for channel id %i' % id)

        while True:
            url = 'http://api.are.na/v2/channels/%s/connections' % id
            payload = {'page':page, 'per':batch_size}
            page += 1

            req = requests.get(url, params=payload)
            channel_connections = req.json()['channels']

            if req.status_code != 200 or len(channel_connections) == 0:
                break

            out.extend([item['id'] for item in channel_connections])

        if len(out) > 1:
            print('Writing channel connections to csv')
            yield {'id':id, 'channel_connections':out}

def write_csv_data(csv_path, data_iterator, target_attrs):
    """
    Opens file from csv_path, and recieves data from data_iterator,
    extracting attributes in target_attrs from json and writing to .csv
    """

    if not os.path.isdir(csv_path.split('/')[0]):
        os.makedirs('./csv')

    if os.path.exists(csv_path):
        os.remove(csv_path)

    f = open(csv_path, 'w+')
    w = csv.writer(f, delimiter=',')

    w.writerow(target_attrs)

    num_written = 0
    ids = set()

    print('Staged to write data to %s' % csv_path)

    for d in data_iterator:
        d['id'] = int(d['id']) #TODO: alter data so all ids are already int

        if d['id'] in ids:
            continue    # if already seen id

        ids.add(d['id'])

        save_data = [value for (key, value) in d.items() if key in target_attrs]

        if len(save_data) == 1: # if from connections TODO: clean up logic
            save_data = save_data[0]

        w.writerow(save_data)
        num_written += 1

    print('Wrote %i rows to %s' % (num_written, csv_path))

    f.close()
    print('Done\n')

def write_channel_csv_to_db(csv_fp, db_fp, table_name):
    conn = sqlite3.connect(db_fp)
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS %s;' % table_name)

    create_table_command = 'CREATE TABLE %s (' % table_name  + \
                                'id int NOT NULL,'           + \
                                'title varchar,'             + \
                                'created_at datetime,'       + \
                                'updated_at datetime,'       + \
                                'published boolean,'         + \
                                'open boolean,'              + \
                                'collaboration boolean,'     + \
                                'slug varchar,'              + \
                                'length int,'                + \
                                'kind varchar,'              + \
                                'status varchar,'            + \
                                'user_id varchar,'           + \
                                'follower_count int,'    + \
                                'contents varchar,'          + \
                                'collaborators varchar'      + \
                            ');'

    c.execute(create_table_command)

    with open(csv_fp) as channels_csv:
        reader = csv.reader(channels_csv)
        channel_data = [tuple(line) for line in reader]

    print('Writing %i rows from %s to %s' % (len(channel_data), csv_fp, db_fp+'/'+table_name))
    for channel in channel_data:
        while len(channel) < len(CHANNEL_TARGET_ATTRS): channel += (None,)
        insert_command = 'INSERT INTO %s VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)' % table_name
        c.execute(insert_command, channel)

    conn.commit()
    print('Done\n')

def write_user_csv_to_db(csv_fp, db_fp, table_name):
    conn = sqlite3.connect(db_fp)
    c = conn.cursor()

    c.execute('DROP TABLE IF EXISTS %s;' % table_name)

    create_table_command = 'CREATE TABLE %s (' % table_name  + \
                                'id int NOT NULL,'           + \
                                'slug varchar,'              + \
                                'username varchar,'          + \
                                'first_name varchar,'        + \
                                'last_name varchar,'         + \
                                'channel_count int,'         + \
                                'following_count int,'       + \
                                'profile_id int,'            + \
                                'follower_count int'         + \
                            ');'

    c.execute(create_table_command)

    with open(csv_fp, 'r') as f:
        reader = csv.reader(f)
        user_data = [tuple(line) for line in reader]

    print('Writing %i rows from %s to %s' % (len(user_data), csv_fp, db_fp+'/'+table_name))

    for user in user_data:
        insert_command = 'INSERT INTO %s VALUES (?,?,?,?,?,?,?,?,?)' % table_name
        c.execute(insert_command, user)

    conn.commit()
    print('Done\n')

def main():
    args = parse_args()

    channel_csv_fp = args.channel_csv
    user_csv_fp = args.user_csv
    block_conn_csv_fp = args.block_conn_csv
    channel_conn_csv_fp = args.channel_conn_csv

    db_fp = args.db
    channel_table_name = args.channel_table
    user_table_name = args.user_table
    batch_size = args.batch_size

    channel_iterator = channel_request_iterator(batch_size)
    write_csv_data(channel_csv_fp, channel_iterator, CHANNEL_TARGET_ATTRS)
    write_channel_csv_to_db(channel_csv_fp, db_fp, channel_table_name)

    user_iterator = user_request_iterator(batch_size)
    write_csv_data(user_csv_fp, user_iterator, USER_TARGET_ATTRS)
    write_user_csv_to_db(user_csv_fp, db_fp, user_table_name)

    block_conn_iterator = block_connections_iterator(channel_csv_fp, batch_size)
    write_csv_data(block_conn_csv_fp, block_conn_iterator, BLOCK_CONN_TARGET_ATTRS)

    channel_conn_iterator = channel_connections_iterator(channel_csv_fp, batch_size)
    write_csv_data(channel_conn_csv_fp, channel_conn_iterator, CHANNEL_CONN_TARGET_ATTRS)

if __name__ == '__main__':
    main()
