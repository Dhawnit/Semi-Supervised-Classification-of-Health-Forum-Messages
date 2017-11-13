import gensim
import logging
import csv
import random
import re
import numpy as np
from collections import defaultdict
from random import shuffle
from sklearn import utils
from gensim.models import doc2vec
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn import metrics


LabeledSentence = gensim.models.doc2vec.LabeledSentence
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def check(category):
	if(category=='SOCL'):
		return 0
	if(category=='DEMO'):
		return 1
	if(category=='FAML'):
		return 2
	if(category=='PREG'):
		return 3
	if(category=='GOAL'):
		return 4
	if(category=='TRMT'):
		return 5
	if(category=='DISE'):
		return 6

def parseTitleQuestion(text):
	text = text.lower()
	text = re.sub(r'https?://(.*?) ', '', text)
	text = re.sub('[^a-z]', ' ', text)
	text = re.sub(' +',' ', text)
	temp = [key for key in text.split() if stopwords[key]!=1]
	return temp

stopwords=defaultdict(int)					#Create stopwords list
with open('StopWords.txt','r') as f:
	for line in f:
		line= line.strip()
		stopwords[line]=1


corpus = []
labels = []

with open("ICHI2016-TrainData.tsv") as tsvfile:
	tsvreader = csv.reader(tsvfile, delimiter="\t")
	firstLine = True	
	for line in tsvreader:
		if(firstLine):
			firstLine = False
			continue
		category = line[0]
		category = check(category)
		labels.append(category)
		text = line[1] +" "+ line[2]		#Title and question combined
		temp = parseTitleQuestion(text)
		corpus.append(temp)

corpus_test = []
labels_test = []

with open("new_ICHI2016-TestData_label.tsv") as tsvfile:
	tsvreader = csv.reader(tsvfile, delimiter="\t")
	for line in tsvreader:
		category = line[0]
		category = check(category)
		labels_test.append(category)
		text = line[1] +" "+ line[2]		#Title and question combined
		temp = parseTitleQuestion(text)
		corpus_test.append(temp)

x_train = corpus
y_train = labels
x_test = corpus_test
y_test = labels_test

# # Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
# # We do this by using the LabeledSentence method. The format will be "TRAIN_i" or "TEST_i" where "i" is
# # a dummy index of the review.

def labelizeMessages(message, label_type):
	labelized = []
	for i,v in enumerate(message):
		label = '%s_%s'%(label_type,i)
		labelized.append(LabeledSentence(v, [label]))
	return labelized


allXs = x_train + x_test
x_train = labelizeMessages(x_train, 'Train')
x_test = labelizeMessages(x_test, 'Test')
allXs = labelizeMessages(allXs, 'All')

# for i in range(10):
# 	print x_train[i]
# 	print x_test[i]
# 	print allXs[i]

# Instantiate Doc2Vec model and build vocab
model = doc2vec.Doc2Vec(size = 400, window = 8, min_count = 1, workers = 4)
# model = doc2vec.Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)

model.build_vocab(allXs)

alpha_val = 0.025		# Initial learning rate
min_alpha_val = 1e-4	# Minimum for linear learning rate decay
passes = 20
alpha_delta = (alpha_val - min_alpha_val) / (passes - 1)

#Pass through the data set multiple times, shuffling the training messages each time to improve accuracy
for epoch in range(passes):
	model.alpha, model.min_alpha = alpha_val, alpha_val
	model.train(utils.shuffle(allXs), total_examples=model.corpus_count, epochs=epoch)
	alpha_val -= alpha_delta

model.save('Model_after_train')

model = doc2vec.Doc2Vec.load('Model_after_train')

# print model.docvecs['All_0']
# print model.docvecs['All_9000']

# get training set vectors from our models
def getVecs(model, corpus, size, vecs_type):
	vecs = np.zeros((len(corpus), size))
	for i in range(0 , len(corpus)):
		index = i
		if(vecs_type == 'Test'):
			index = index + 8000
		prefix = 'All_' + str(index)
		# print model.docvecs[prefix]
		vecs[i] = model.docvecs[prefix]
	return vecs

# get train vectors
train_vecs = getVecs(model, x_train, 400, 'Train')
print train_vecs.shape

# get test vectors
test_vecs = getVecs(model, x_test, 400, 'Test')
print test_vecs.shape


# train classifier
# lr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)
lr = LogisticRegression(multi_class='multinomial', solver='newton-cg')
lr.fit(train_vecs, np.array(y_train))

# print "Multinomial Logistic regression Train Accuracy :: ", metrics.accuracy_score(np.array(y_train), lr.predict(train_vecs))
# print "Multinomial Logistic regression Test Accuracy :: ", metrics.accuracy_score(np.array(y_test), lr.predict(test_vecs))

# print lr.predict(test_vecs)

print 'Train Accuracy: %.2f'%lr.score(train_vecs, np.array(y_train))
print 'Test Accuracy: %.2f'%lr.score(test_vecs, np.array(y_test))
