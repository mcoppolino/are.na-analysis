import requests
import sqlite3

DATABASE_PATH = 'data.db'
TARGET_ATTRS = ['id', 'title', 'length']
PER_PAGE = 100

def channel_request_iterator():
    """
    yields a list of channel json data of length PER_PAGE
    """

    page = 2
    while True:
        url = 'http://api.are.na/v2/channels?page=%d&amp;per=%d' % (page, PER_PAGE)
        print(page, url)
        page += 1

        req = requests.get(url)

        if req.status_code == 200:
            yield req.json()['channels']
        else:
            break

def fetch_and_write_channel_data():
    """
    Opens a connection to DATABASE_PATH, and recieves channel data from
    channel_request_iterator, extracting attributes in TARGET_ATTRS from json
    and writing to database
    """
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()

    c.execute('DROP TABLE IF EXISTS channels;')

    c.execute('CREATE TABLE channels (id int NOT NULL, title varchar, length int);')

    for channel_list in channel_request_iterator(): # yield list of json objects
        for channel in channel_list: # for each json object in list

            keep_data = [value for (key, value) in channel.items() if key in TARGET_ATTRS]
            c.execute('INSERT INTO channels VALUES (?, ?, ?)', tuple(keep_data))
            conn.commit()

def main():
    fetch_and_write_channel_data()

if __name__ == '__main__':
    main()
