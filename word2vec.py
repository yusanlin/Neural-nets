"""
word2vec.py
@description: This is an implementation of word2vec using Keras
"""

# imports
import csv
import sys
import re
from collections import Counter
from nltk.stem.porter import PorterStemmer
import nltk

# Some constants that will be refer to later
N_V   = 1000 # number of vocabularies
N_H   = 50   # number of neurons in the hidden layer
C     = 2    # context window size, meaning including C left words and C right words (2C + 1 in total)
MIN_F = 100

# read in data
path_data = "data/"
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
corpus = corpus.lower()           # convert all the letters into lowercases
#wnl = WordNetLemmatizer()
corpus = corpus.decode("ascii", "ignore")

stemmer   = PorterStemmer()
total     = len(corpus.split(" "))
processed = 0.0

# create vocabularies
counter = Counter(corpus.split(" "))
#vocab   = [x for x,y in counter.most_common() if y >= MIN_F]

# get all the unique words and sort them alphabetically
# add in the special token for buffer as well
vocab   = [x for x,y in counter.most_common()]; vocab = vocab.append("<s>"); vocab.sort()
N_V     = len(vocab)
# corpus  = (" ").join([x for x in corpus.split(" ") if x in vocab])

# split the corpus into sentences
corpus = nltk.tokenize.sent_tokenize(corpus)

# after tokenizing as sentences, remove all the symbols
corpus = [re.sub(r'[^\w]', ' ', s) for s in corpus]

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

# Training
for sentence in corpus:

	sentence_tmp = "<s> " * C + sentence + " <s>" * C

	# Everytime send in a context window
	for i in range(C, len(sentence_tmp) - C + 1):
		x_train_batch = [0] * N_V

		for j in range(i - C, i + C + 1):
			if i != j:
				tmp = [0] * N_V
				tmp[vocab.index(sentence_tmp[j])] = 1
				x_train_batch += tmp

		x_train_batch = x_train_batch / float(2 * C)

		# the ith word is thus the target word for prediction
		y_train_batch = [0] * N_V; y_train_batch[vocab.index(sentence_tmp[i])] = 1

		# use the above as the input for the input for the model
		model.fit(x_train_batch, y_train_batch)
