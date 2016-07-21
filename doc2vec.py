"""
doc2vec.py
@author: Yusan Lin
@description: This program implements the Paragraph Vector proposed in 
[1] Quoc Le, Tomas Mikolov, Distributed Representations of Sentences and Documents, https://cs.stanford.edu/~quocle/paragraph_vector.pdf
"""

from initializer import *
from functions import *
from loader import *

# ---------------
# Load the corpus
# ---------------

print "Processing the corpus data..."
corpus, vocab = load_corpus("unlabeledTrainData.tsv")

# -----------------------------------------
# Build the model (different from word2vec)
# -----------------------------------------
N = len(corpus) # number of paragraphs/docs
M = len(vocab)  # number of vocabularies

p = 10          # dimension of paragraphs
q = 50          # dimension of words

model_p = Sequential()
model_p.add(Dense(output_dim = p, input_dim = N)) # the first layer is for paragraph vector
model_p.add(Activation("linear"))

model_w = Sequential()
model_w.add(Dense(output_dim = q, input_dim = M))
model_w.add(Activation("linear"))

merged_model = Sequential()
merged_model.add(Merge([model_p, model_w], mode = 'concat', concat_axis = 1))
merged_model.add(Activation("softmax"))

# Compile the model
merged_model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

# ------------------------------------------------
# Sampling the data (also different from word2vec)
# ------------------------------------------------

# Beacuse we don't want to try on the whole dataset yet, let's subsample it
print "There are", len(corpus), "paraphs in the dataset."
n_samples = int(raw_input("How many paragraphs do you want to sub sample? "))
corpus_subsample = random.sample(corpus, n_samples)

X = []
Y = []

processed_paragraphs = 0 

# for the same paragrah, they receive the same paragraph id
pid = 0 
for paragraph in corpus_subsample:
	d = np.zeros(N) # initialize a single paragraph vector with N zeros
	d[pid] = 1

	print processed_paragraphs, "paragraphs processed"

	# do context window over a paragraph
	for sentence in nltk.tokenize.sent_tokenize(paragraph):

		#print processed_sentences, "sentences processed"
		#print sentence

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

			X.append(list(d) + list(x)) # input is the concatenation of paragraph and words
			Y.append(list(y))		

	processed_paragraphs += 1

X = np.array(X)
Y = np.array(Y)

# Split the X and Y into training and testing
print "Splitting into training and testing"
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)

# Fit the model
print "Fitting the model"
accuracy = float("-inf")
while accuracy < 50 or interations < MAX_ITER:
	merged_model.fit(X_train, y_train, verbose = 0)
	accuracy = merged_model.evaluate(X_test, y_test)
	accuracy = accuracy[1]
	print "Accuracy: ", accuracy, "%"

