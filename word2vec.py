"""
word2vec.py
@author: Yusan Lin
@description: This is an implementation of word2vec using Keras
"""

from initializer import *
from sampling import *
from loader import *

# Some constants that will be refer to later
N_V      = 1000 # number of vocabularies
N_H      = 50   # number of neurons in the hidden layer

# ---------------
# Load the corpus
# ---------------

print "Processing the corpus data..."
corpus, vocab = load_corpus("unlabeledTrainData.tsv")

# We start from building a model container
print "Building model..."
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

# ------------------------------------------------
# Sampling the data (also different from word2vec)
# ------------------------------------------------
print "Cutting training and testing..."
# Beacuse we don't want to try on the whole dataset yet, let's subsample it
n_samples = int(raw_input("How many sentences do you want to sub sample? "))
corpus_subsample = random.sample(corpus, n_samples)
corpus = None # discard corpus

# split the corpus into training and testing
random.shuffle(corpus_subsample)
test_size = 0.33
corpus_training = corpus_subsample[:int((1-test_size)*len(corpus_subsample))]
corpus_testing  = corpus_subsample[int((1-test_size)*len(corpus_subsample)):]

# Sample the selected corpus to form X and Y
#X, Y = sample_paragraph(corpus_subsample, vocab)

# Split the X and Y into training and testing
#print "Splitting into training and testing"
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)

# Fit the model
print "Fitting the model"
training_step_size = 100 # how many sentences I want to train in each iteration
iteration = 0
while accuracy < 50 or interations < MAX_ITER:
	print "Iteration", iteration
	# loop through the corpus_training and fit a little bit at a time
	for i in xrange(0, len(corpus_training), training_step_size):
		tmp_train = corpus_training[i : i + training_step_size]
		tmp_test  = corpus_testing[i : i + training_step_size]

		# produce X and y
		X_train, y_train = sample_word(tmp_train, vocab)
		X_test,  y_test  = sample_word(tmp_test,  vocab)

		model.fit(X_train, y_train, verbose = 0)
		#model.fit_generator(sample_word(corpus_subsample, vocab), samples_per_epoch=10000, nb_epoch=10)
		accuracy = model.evaluate(X_test, y_test)
		accuracy = accuracy[1]
		print "Accuracy: ", accuracy, "%"

		iteration += 1
