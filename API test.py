import requests
import sqlite3
import json

base_url = "http://api.are.na/v2/"

#slug thumb:
#channel = "arena-influences"
#call = base_url + "channels/" + channel + "/thumb"

#channel search :#
query = "pokemon"
call = base_url + "search?q=" + query

req = requests.get(call)
j_data = req.json()
data = json.dumps(j_data)
parsed = json.loads(data)
#print(json.dumps(parsed, indent=2, sort_keys=True))
blocks = parsed['blocks']
channels = parsed['channels']
for i in blocks:
    print(i['title'])
#print(parsed['blocks'])
