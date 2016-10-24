import json
import sqlite3
json_file = "b_id_state.json"
db_file = "data.db"
new_file = open("Pittsburgh_business.json", 'a')
read_file = open(json_file, 'r')
lines = read_file.readlines()
false = 0
true = 1
city = "Pittsburgh"
a =[]
for line in lines:
	line = line.strip('\n')

	line = eval(line)

	if line['city'] == city:
		recode = line['business_id'] + '\n'
		new_file.write(recode)
