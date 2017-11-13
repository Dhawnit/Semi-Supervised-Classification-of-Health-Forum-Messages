from __future__ import print_function

import gensim
import logging
import csv
import random
from collections import defaultdict
import re
import random
import numpy as np
from random import shuffle
from sklearn import utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from gensim.models import doc2vec

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.models import load_model
from keras.layers import Convolution1D, GlobalAveragePooling1D, MaxPooling1D, Flatten, Bidirectional, Activation


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
	return ' '.join(temp) #for tf-idf vector
	# return temp  #for Doc2vec

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


# Using doc2vec for getting document vectors
# def labelizeMessages(message, label_type):
# 	labelized = []
# 	for i,v in enumerate(message):
# 		label = '%s_%s'%(label_type,i)
# 		labelized.append(LabeledSentence(v, [label]))
# 	return labelized

# allXs = x_train + x_test
# x_train = labelizeMessages(x_train, 'Train')
# x_test = labelizeMessages(x_test, 'Test')
# allXs = labelizeMessages(allXs, 'All')

# Loading Doc2Vec Model
# model = doc2vec.Doc2Vec.load('Model_after_train')

# def getVecs(model, corpus, size, vecs_type):
# 	vecs = np.zeros((len(corpus), size))
# 	for i in range(0 , len(corpus)):
# 		index = i
# 		if(vecs_type == 'Test'):
# 			index = index + 8000
# 		prefix = 'All_' + str(index)
# 		# print model.docvecs[prefix]
# 		vecs[i] = model.docvecs[prefix]
# 	return vecs

# # get train vectors
# train_vecs = getVecs(model, x_train, 800, 'Train')
# print (train_vecs.shape)

# test_vecs = getVecs(model, x_test, 800, 'Test')
# print (test_vecs.shape)

# Getting document vectors using count vectorizer and tf-idf 
# count vectorizer
count_vect = CountVectorizer()
data_count = count_vect.fit_transform(x_train + x_test)
print (data_count.shape)


# tfidf model
tfidf_transformer = TfidfTransformer()
data_tfidf = tfidf_transformer.fit_transform(data_count)
print (data_tfidf.shape)

data_tfidf = data_tfidf.astype('float32')

# Dimensionality reduction
print ('PCA Start')
svd = TruncatedSVD(n_components=500, n_iter=7, random_state=42)
svd.fit(data_tfidf)
data_tfidf = svd.transform(data_tfidf)
print ('PCA Done')

print (data_tfidf.shape)

train_vecs = data_tfidf[:8000]
test_vecs = data_tfidf[8000:]

print (train_vecs.shape)
print (test_vecs.shape)

batch_size = 64
max_features = 500
num_classes = 7
epochs = 30

# Loss Increases if comment below 4 lines but good test accuracy
# train_vecs = train_vecs.astype('float32')
# test_vecs = test_vecs.astype('float32')
# train_vecs /= 255
# test_vecs /= 255


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(y_train.shape)
print(y_test.shape)

## DEEP NEURAL NETWORKS

model = Sequential()
model.add(Dense(250, activation='relu', input_shape=(max_features,)))
model.add(Dropout(0.5))
model.add(Dense(125, activation='relu'))
model.add(Dropout(0.5))
# model.add(Dense(62, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

## LSTM 1

# model = Sequential()
# model.add(Embedding(max_features, 128))
# model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(num_classes, activation='sigmoid'))


## LSTM 2

# train_vecs = np.reshape(train_vecs,(train_vecs.shape[0],train_vecs.shape[1],1))
# test_vecs = np.reshape(test_vecs,(test_vecs.shape[0],test_vecs.shape[1],1))
# model = Sequential()
# model.add(LSTM(128, input_shape =(train_vecs.shape[1],train_vecs.shape[2])))
# model.add(Activation('relu'))
# model.add(Dense(64, init='uniform'))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(7))
# model.add(Activation('softmax'))


model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',  #RMSprop()
              metrics=['accuracy'])

history = model.fit(train_vecs, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.2
                    )

score = model.evaluate(test_vecs, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


model.save('my_model.h5') # creates a HDF5 file 'my_model.h5'
# del model  # deletes the existing model

# returns a compiled model identical to the previous one
# model = load_model('my_model.h5')


## Experiments
# Batch sizes -> 16, 32, 64(.64 accuracy), 100(.64 accuracy)
# Layers Configuartion Best 500->250->125->7 