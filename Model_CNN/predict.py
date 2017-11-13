import os
import sys
import json
import re
import logging
from collections import Counter
import csv
import random
from collections import defaultdict
from random import shuffle
import data_helper
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

logging.getLogger().setLevel(logging.INFO)

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
		for line in tsvreader:
			category = line[0]
			labels.append(category)
			text = line[1] +" "+ line[2]		#Title and question combined
			temp = parseTitleQuestion(text)
			corpus.append(temp)

	labelsSet = sorted(list(set(labels)))
	x_raw = corpus
	y_raw = []
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
	
	return x_raw, y_raw

def predict_unseen_data():

	"""Step 0: load trained model and parameters"""
	params = json.loads(open('./parameters.json').read())
	checkpoint_dir = sys.argv[1]
	if not checkpoint_dir.endswith('/'):
		checkpoint_dir += '/'
	checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir + 'checkpoints')
	logging.critical('Loaded the trained model: {}'.format(checkpoint_file))
	
	"""Step 1: load data for prediction"""
	test_file = sys.argv[2]
	x_test, y_test = load_data_and_labels(test_file)
	
	# x_test = []
	# with open("Unspervised.txt") as f:
	# 	for line in f:
	# 		x_test.append(line)

	# labels.json was saved during training, and it has to be loaded during prediction
	labels = json.loads(open('./labels.json').read())

	vocab_path = os.path.join(checkpoint_dir, "vocab.pickle")
	vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
	x_test = np.array(list(vocab_processor.transform(x_test)))
	
	"""Step 2: compute the predictions"""
	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)

		with sess.as_default():
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)

			input_x = graph.get_operation_by_name("input_x").outputs[0]
			dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
			predictions = graph.get_operation_by_name("output/predictions").outputs[0]

			batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)
			all_predictions = []
			for x_test_batch in batches:
				batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
				all_predictions = np.concatenate([all_predictions, batch_predictions])
	
	# all_predictions = all_predictions.tolist()
	# for x in all_predictions:
	# 	print(x)

	if y_test is not None:
		y_test = np.argmax(y_test, axis=1)
		correct_predictions = sum(all_predictions == y_test)
		logging.critical('The accuracy is: {}'.format(correct_predictions / float(len(y_test))))
		logging.critical('The prediction is complete')
		all_predictions = all_predictions.tolist()
		with open('output.txt','w') as file:
			ret = []
			for x in all_predictions:
				ret.append(labels[int(x)])
			file.write(str(ret))
			print(len(ret))

if __name__ == '__main__':
	predict_unseen_data()
