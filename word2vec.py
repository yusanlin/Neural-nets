"""
word2vec.py
@author: Yusan Lin
@description: This is an implementation of word2vec using Keras
"""

# imports
import re
import csv
import sys
import nltk
import pickle
import random
import decimal
import os.path

import numpy as np

from collections import Counter
from nltk.stem.porter import PorterStemmer
from sklearn.cross_validation import train_test_split

# Some constants that will be refer to later
N_V      = 1000 # number of vocabularies
N_H      = 50   # number of neurons in the hidden layer
C        = 2    # context window size, meaning including C left words and C right words (2C + 1 in total)
MIN_F    = 100
MAX_ITER = 10   # maximum allowed training iterations

path_data = "../data/"

# check whether the processed corpus exists
if os.path.isfile(path_data + "imdb_corpus.p"):
	print "Loading in the processed corpus data..."
	corpus = pickle.load(open(path_data + "imdb_corpus.p", "rbU"))
	vocab  = pickle.load(open(path_data + "imdb_vocab.p", "rbU"))
	N_V    = len(vocab)

else:
	print "Processing the corpus data..."
	# read in data
	filename  = "unlabeledTrainData.tsv"
	corpus = "" # initialize corpus into a long long string that concatenates all the reviews together
	with open(path_data + filename, "rbU") as f:
		next(f) # skip the first line
		reader = csv.reader(f, delimiter = '\t')
		for row in reader:
			corpus += row[1] + " "

	# clean the corpus
	corpus = corpus.replace("<br />", "") # remove all the new line tags
	corpus = corpus.replace("\\", "")     # remove all the annoying back slashes
	corpus = corpus.lower()               # convert all the letters into lowercases
	#wnl = WordNetLemmatizer()
	corpus = corpus.decode("ascii", "ignore")

	stemmer   = PorterStemmer()
	total     = len(corpus.split(" "))
	processed = 0.0
	TWOPLACES = decimal.Decimal(10) ** -2

	# corpus = " ".join([stemmer.stem(x) for x in corpus.split(" ")])
	corpus_stemmed = []
	for x in corpus.split(" "):
		sys.stdout.write("\r{0}%".format(decimal.Decimal(processed / total * 100).quantize(TWOPLACES))) 
		sys.stdout.flush()
		corpus_stemmed.append(stemmer.stem(x))
		processed += 1

	corpus = " ".join(corpus_stemmed)
	corpus_stemmed = None

	# create vocabularies
	counter = Counter(corpus.split(" "))
	#vocab   = [x for x,y in counter.most_common() if y >= MIN_F]

	# get all the unique words and sort them alphabetically
	# add in the special token for buffer as well
	vocab   = [x for x,y in counter.most_common()]; vocab.append("<s>"); vocab.sort()
	N_V     = len(vocab)
	# corpus  = (" ").join([x for x in corpus.split(" ") if x in vocab])

	# split the corpus into sentences
	corpus = nltk.tokenize.sent_tokenize(corpus)

	# after tokenizing as sentences, remove all the symbols
	corpus = [re.sub(r'[^\w]', ' ', s) for s in corpus]

	# save up the corpus after preprocessing. Only needs to load it in the future
	pickle.dump(corpus, open(path_data + "imdb_corpus.p", "w"))
	pickle.dump(vocab,  open(path_data + "imdb_vocab.p", "w"))

# We start from building a model container
from keras.models import Sequential
model = Sequential()

# We then stack the layers
from keras.layers import Dense, Activation
model.add(Dense(output_dim = N_H, input_dim = N_V))
model.add(Activation("linear"))
model.add(Dense(output_dim = N_V))
model.add(Activation("softmax"))

# Compile the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

total_sentences   = len(corpus)
processed_sentences = 0.0

accuracy = float("-inf")

# Beacuse we don't want to try on the whole dataset yet, let's subsample it
n_samples = int(raw_input("How many sentences do you want to sub sample? "))
corpus_subsample = random.sample(corpus, n_samples)

#X = np.array()
#y = np.array()
X = []
Y = []

# Construct X and Y
print "Constructing X and Y"
for sentence in corpus_subsample:

	print processed_sentences, "sentences processed"
	print sentence

	sentence_tmp = "<s> " * C + sentence + " <s>" * C
	sentence_tmp = sentence_tmp.split()

	# Everytime send in a context window
	for i in range(C, len(sentence_tmp) - C + 1):
		#print sentence_tmp[i]
		x = np.zeros(N_V)

		for j in range(i - C, i + C):
			if i != j:
				tmp = np.zeros(N_V)
				try:
					tmp[vocab.index(sentence_tmp[j])] = 1
				except ValueError:
					pass
				x += tmp

		x = np.array([ w/float(2 * C) for w in x]).reshape((1, N_V))[0]

		# the ith word is thus the target word for prediction
		y = np.zeros(N_V)
		try:
			y[vocab.index(sentence_tmp[i])] = 1
		except ValueError:
			pass
		y = y.reshape((1, N_V))[0]

		X.append(list(x))
		Y.append(list(y))

	processed_sentences += 1

X = np.array(X)
Y = np.array(Y)

# Split the X and Y into training and testing
print "Splitting into training and testing"
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)

# Fit the model
print "Fitting the model"
while accuracy < 50 or interations < MAX_ITER:
	model.fit(X_train, y_train, verbose = 0)
	accuracy = model.evaluate(X_test, y_test)
	accuracy = accuracy[1]
	print "Accuracy: ", accuracy, "%"
