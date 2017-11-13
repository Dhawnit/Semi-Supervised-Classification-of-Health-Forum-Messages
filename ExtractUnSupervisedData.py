import os
import sys
from collections import defaultdict
import re

def parseTitleQuestion(text):
	text = text.lower()
	text = re.sub(r'https?://(.*?) ', '', text)
	text = re.sub('[^a-z]', ' ', text)
	text = re.sub(' +',' ', text)
	temp = [key for key in text.split() if stopwords[key]!=1]
	temp = ' '.join(temp)
	return temp

stopwords=defaultdict(int)					#Create stopwords list
with open('StopWords.txt','r') as f:
	for line in f:
		line= line.strip()
		stopwords[line]=1

corpus = []
for fil in os.listdir('contains_all/'):
	fil = 'contains_all/'+ fil
	with open(fil) as f:
		for line in f:
			line=line.split('<<->>')
			corpus.append(parseTitleQuestion(line[0] + " " +line[1]))

print(len(corpus))
for line in corpus:
	print(line)