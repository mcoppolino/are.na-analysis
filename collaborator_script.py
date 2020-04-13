import requests
import sqlite3
import csv, json, sys

base_url = "http://api.are.na/v2/channels/"
end_url = "/collaborators"

with open('channels.csv', mode='r') as channels.csv:
    csv_reader = csv.reader(channels.csv, delimiter=',')
    for row in csv_reader:
        channel_name = row['title']
        channel_id = row['id']
        call = base_url + channel_id + end_url
        req = requests.get(call)
        j_data = req.json()
        data = json.dumps(j_data)
        parsed = json.loads(data)
        users = parsed['users']
        print(users)
def collaborator_iterator(channels_csv_fp):

    base_url = "http://api.are.na/v2/channels/"
    end_url = "/collaborators"

    with open('collaborator.csv', mode='w') as fw:
        fieldnames = ['channel', 'collaborator']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        with open(channels_csv_fp, mode='r') as f:
            reader = csv.reader(f)
            channel_ids = [tuple(line)[0] for line in reader]

        for id in channel_ids[1:]:
            call = base_url + id + end_url
            req = requests.get(call)
            j_data = req.json()
            data = json.dumps(j_data)
            parsed = json.loads(data)
            users = parsed['users']
            for user in users:
                writer.writerow({'channel': id, 'collaborator': user['id']})

#for i in users:
    #print(i['id'])
#channels = parsed['channels']
#for i in blocks:
#    print(i['title'])
#print(parsed['blocks'])
