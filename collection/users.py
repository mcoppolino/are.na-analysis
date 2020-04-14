import argparse
import csv
import requests
import sqlite3
from utils import write_csv_data


# the user attributes to be collected from API call GET /v2/search (returns users)
# https://dev.are.na/documentation/search#Block59799
USER_TARGET_ATTRS = ['id', 'slug', 'username', 'first_name', 'last_name',
                     'channel_count', 'following_count', 'profile_id', 'follower_count']


def parse_args():
    parser = argparse.ArgumentParser(description='are.na data collection')

    parser.add_argument('-user_csv', help='path to user csv file to be written to', default='data/csv/users.csv')
    parser.add_argument('-user_table', help='user table name in database', default='users')
    parser.add_argument('-db', help='path to db file to be written to', default='data/data.db')
    parser.add_argument('-batch_size', help='batch_size for requests (as large as possible for speed)', default=10000)

    return parser.parse_args()


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
            payload = {'q': letter, 'page': page, 'per': batch_size}

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


def write_user_csv_to_db(csv_fp, db_fp, table_name):
    conn = sqlite3.connect(db_fp)
    c = conn.cursor()

    c.execute('DROP TABLE IF EXISTS %s;' % table_name)

    create_table_command = 'CREATE TABLE %s (' % table_name + \
                           'id int NOT NULL,' + \
                           'slug varchar,' + \
                           'username varchar,' + \
                           'first_name varchar,' + \
                           'last_name varchar,' + \
                           'channel_count int,' + \
                           'following_count int,' + \
                           'profile_id int,' + \
                           'follower_count int' + \
                           ');'

    c.execute(create_table_command)

    with open(csv_fp, 'r') as f:
        reader = csv.reader(f)
        user_data = [tuple(line) for line in reader]

    print('Writing %i rows from %s to %s' % (len(user_data), csv_fp, db_fp + '/' + table_name))

    for user in user_data:
        insert_command = 'INSERT INTO %s VALUES (?,?,?,?,?,?,?,?,?)' % table_name
        c.execute(insert_command, user)

    conn.commit()
    print('Done\n')


def main():
    args = parse_args()

    user_csv_fp = args.user_csv
    db_fp = args.db
    user_table_name = args.user_table
    batch_size = args.batch_size

    user_iterator = user_request_iterator(batch_size)
    write_csv_data(user_csv_fp, user_iterator, USER_TARGET_ATTRS)
    write_user_csv_to_db(user_csv_fp, db_fp, user_table_name)


if __name__ == '__main__':
    main()
