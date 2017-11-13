import re
import logging
import numpy as np
from collections import Counter
import csv
import random
from collections import defaultdict
from random import shuffle


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


#Load sentences and labels
def load_data_and_labels(filename):
	corpus = []
	labels = []
	with open(filename) as tsvfile:
		tsvreader = csv.reader(tsvfile, delimiter="\t")
		firstLine = True	
		for line in tsvreader:
			if(firstLine):
				firstLine = False
				continue
			category = line[0]
			labels.append(category)
			text = line[1] +" "+ line[2]		#Title and question combined
			temp = parseTitleQuestion(text)
			corpus.append(temp)

	labelsSet = sorted(list(set(labels)))
	x_raw = corpus
	y_raw = []

	#one hot encoding
	for x in labels:
		temp = [ 0 for i in range(0,len(labelsSet))]
		cnt = 0
		for elem in labelsSet:
			if elem == x:
				temp[cnt] = 1
				break
			cnt += 1
		temp = np.array(temp)
		y_raw.append(temp)
	
	return x_raw, y_raw, labelsSet

# Iterate the data batch by batch
def batch_iter(data, batch_size, num_epochs, shuffle=True):
	
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(data_size / batch_size) + 1

	for epoch in range(num_epochs):
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]
		
if __name__ == '__main__':
	input_file = 'ICHI2016-TrainData.tsv'
	load_data_and_labels(input_file)
