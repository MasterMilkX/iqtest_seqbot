import json
import sys

#load json
with open(sys.argv[1]) as f:
	data = json.load(f)

#add hint
for d in data.keys():
	data[d]["hint"] = ""

#print it back out
print(data)