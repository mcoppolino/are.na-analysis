import requests
import sqlite3
import csv, json, sys

base_url = "http://api.are.na/v2/"

#slug thumb:
#channel = "arena-influences"
#call = base_url + "channels/" + channel + "/thumb"

#channel search :#
query = "pokemon"
call = base_url + "search?q=" + query

call = "http://api.are.na/v2/channels/arena-influences/collaborators"

req = requests.get(call)
j_data = req.json()
data = json.dumps(j_data)
parsed = json.loads(data)
#print(json.dumps(parsed, indent=2, sort_keys=True))
users = parsed['users']
print(users)
#for i in users:
    #print(i['id'])
#channels = parsed['channels']
#for i in blocks:
#    print(i['title'])
#print(parsed['blocks'])
