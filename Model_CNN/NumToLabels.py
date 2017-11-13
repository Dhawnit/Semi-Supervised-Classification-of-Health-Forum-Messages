import os
import sys
import json

labels = json.loads(open('./labels.json').read())
x = []
with open("OutputSemiSupervised.txt") as f:
	for line in f:
		x.append(int(float(line.strip())))
for n in x:
	print(labels[n])
	
