import requests
import sqlite3

TARGET_ATTRS = ['id', 'title', 'length']
NUM_PAGES = 10

def fetch_channel_data():
    channels = []

    for i in range(2, NUM_PAGES + 2):
        url = 'http://api.are.na/v2/channels?page=%d&amp;per=10' % i
        print(i, url)

        channel = requests.get(url).json()['channels']
        channels.extend(channel)

    return channels

def write_channel_data(channels):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()

    c.execute('DROP TABLE IF EXISTS channels;')

    c.execute('CREATE TABLE channels (id int NOT NULL, title varchar, length int);')

    for channel in channels:
        keep_data = [value for (key, value) in channel.items() if key in TARGET_ATTRS]
        c.execute('INSERT INTO channels VALUES (?, ?, ?)', tuple(keep_data))

    conn.commit()


def main():
    channels = fetch_channel_data()
    write_channel_data(channels)

if __name__ == '__main__':
    main()
