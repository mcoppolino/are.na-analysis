import argparse
import csv
import os
import requests
import sqlite3
import sys
from string import ascii_lowercase

# the attributes to be collected from API call GET /v2/channels
# https://dev.are.na/documentation/channels#Block43473
CHANNEL_TARGET_ATTRS = ['id', 'title', 'created_at', 'updated_at', 'published',
    'open', 'collaboration', 'slug', 'length', 'kind', 'status', 'user_id', \
    'follower_count', 'contents', 'collaborators']

# the user attributes to be collected from API call GET /v2/search (returns users)
# https://dev.are.na/documentation/search#Block59799
USER_TARGET_ATTRS = ['id', 'slug', 'username', 'first_name', 'last_name', \
    'channel_count', 'following_count', 'profile_id', 'follower_count']

def parse_args():
    parser = argparse.ArgumentParser(description='are.na data collection')
    parser.add_argument('-c', help='path to channel csv file to be written to', default='csv/channels.csv')
    parser.add_argument('-u', help='path to user csv file to be written to', default='csv/users.csv')
    parser.add_argument('-d', help='path to db file to be written to', default='data.db')
    parser.add_argument('-ct', help='channel table name in database', default='channels')
    parser.add_argument('-ut', help='user table name in database', default='users')
    parser.add_argument('-b', help='batch_size for requests (as large as possible for speed)', default=10000)

    return parser.parse_args()

def channel_request_iterator(batch_size):
    """
    yields a list of channel json data of length batch_size
    """

    print('Requesting channel data from are.na API')

    page = 2
    while True:
        url = 'http://api.are.na/v2/channels'
        payload = {'page':page, 'per':batch_size}
        page += 1

        req = requests.get(url, params=payload)
        channel_data = req.json()['channels']

        if req.status_code != 200 or len(channel_data) == 0:
            break

        yield channel_data

def user_request_iterator(batch_size):
    """
    yields a list of user json data of length batch_size
    """

    print('Requesting user data from are.na API')

    for letter in 'abcdefghijklmnopqrstuvwxyz':
        page = 1
        print('Fetching users with query "%s"' % letter)
        while True:
            url = 'http://api.are.na/v2/search/users/'
            payload = {'q':letter, 'page':page, 'per':batch_size}
            page += 1

            req = requests.get(url, params=payload)

            user_data = req.json()['users']

            if req.status_code != 200 or len(user_data) == 0:
                break

            yield user_data

def write_csv_data(csv_path, data_list_iterator, target_attrs):
    """
    Opens file from csv_path, and recieves data from data_list_iterator,
    extracting attributes in target_attrs from json and writing to .csv
    """

    if os.path.exists(csv_path):
        os.remove(csv_path)

    f = open(csv_path, 'w+')
    w = csv.writer(f, delimiter=',')

    w.writerow(target_attrs)

    num_written = 0
    ids = set()
    for data_list in data_list_iterator: # yield list of json objects
        for d in data_list: # for each json object in list
            if d['id'] in ids:
                continue    # if already seen id

            ids.add(d['id'])

            save_data = [value for (key, value) in d.items() if key in target_attrs]
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
                                'follower_count varchar,'    + \
                                'contents varchar,'          + \
                                'collaborators varchar'      + \
                            ');'

    c.execute(create_table_command)

    csv.field_size_limit(sys.maxsize)
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

    with open(csv_fp, mode='r') as f:
        reader = csv.reader(f)
        user_data = [tuple(line) for line in reader]

    print('Writing %i rows from %s to %s' % (len(user_data), csv_fp, db_fp+'/'+table_name))

    for user in user_data:
        insert_command = 'INSERT INTO %s VALUES (?,?,?,?,?,?,?,?,?)' % table_name
        c.execute(insert_command, user)

    conn.commit()
    print('Done\n')

# def get_user_following()
def main():
    args = parse_args()
    import pdb; pdb.set_trace()
    channel_csv_fp, user_csv_fp, db_fp, channel_table_name, user_table_name, batch_size = args.c, args.u, args.d, args.ct, args.ut, args.b

    channel_iterator = channel_request_iterator(batch_size)
    write_csv_data(channel_csv_fp, channel_iterator, CHANNEL_TARGET_ATTRS)
    write_channel_csv_to_db(channel_csv_fp, db_fp, channel_table_name)

    user_iterator = user_request_iterator(batch_size)
    write_csv_data(user_csv_fp, user_iterator, USER_TARGET_ATTRS)
    write_user_csv_to_db(user_csv_fp, db_fp, user_table_name)


if __name__ == '__main__':
    main()
