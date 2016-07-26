"""
word2vec.py
@author: Yusan Lin
@description: This version of word2vec is designed in class form
"""

from keras.models import Sequential
from keras.layers import Dense, Activation

class Word2vec(object):
	"""A word2vec model.

	Attributes:
	    vocab: The vocabulary of words that will later refer to
	    N_H  : The dimension of the hidden layer, which is also the dimension of the 
	           lower-dimensional representation of word vectors
	    model: The model built using Keras
	"""

	def __init__(self, vocab, N_H):
		"""Return a Word2vec object using vocab as the input dimension and N_H as the 
		   dimension for hidden layer
		"""
		# Some constants
		self.vocab = vocab
		self.N_V   = len(vocab)
		self.N_H   = N_H

		# Create the Keras model
		self.model = Sequential()
		self.model.add(Dense(output_dim = N_H, input_dim = N_V))
		self.model.add(Activation("linear"))
		self.model.add(Dense(output_dim = N_V))
		self.model.add(Activation("softmax"))
		self.model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', 
			metrics = ['accuracy'])


	def sample_a_sent(self, sent, C):
		"""Sample the given sentence into multiple context windows with window size 2C. 
		   Return the corresponding X (input for the network) and Y (target for the network).
		   Both X and Y are returned in lists instead of numpy arrays.
		"""
		X = []
		Y = []

		sent_tmp = "<s> " * C + sent + " <s>" * C
		sent_tmp = sent_tmp.split()

		for i in range(C, len(sent_tmp) - C + 1):
			#print sent_tmp[i]
			x = np.zeros(N_V)

			for j in range(i - C, i + C):
				if i != j:
					tmp = np.zeros(N_V)
					try:
						tmp[vocab.index(sent_tmp[j])] = 1
					except ValueError:
						pass
					x += tmp

			x = np.array([ w/float(2 * C) for w in x]).reshape((1, N_V))[0]

			# the ith word is thus the target word for prediction
			y = np.zeros(N_V)
			try:
				y[vocab.index(sent_tmp[i])] = 1
			except ValueError:
				pass
			y = y.reshape((1, N_V))[0]

			X.append(list(x))
			Y.append(list(y))

		return X, Y


	def sample_sents(self, sents, C):
		"""Sample the given list of sentences. 
		   Return the X and Y in numpy arrays.
		"""	
		X = []
		Y = []

		for sent in sents:
			X_tmp, Y_tmp = sample_a_sent(sent, C)
			
			X.extend(X_tmp)
			Y.extend(Y_tmp)
			
			X_tmp = None
			Y_tmp = None

		return np.array(X), np.array(Y)


	def train(corpus_training, corpus_testing, 
		training_step_size = 100, max_iter = 100, min_accu = 0.5, C = 5):
		"""Train the word2vec model.
		"""
		iteration = 0
		accuracy = float("-inf")

		while accuracy < min_accu or interations < max_iter:
			print "Iteration", iteration

			# loop through the corpus_training and fit a little bit at a time
			for i in xrange(0, len(corpus_training), training_step_size):
				tmp_train = corpus_training[i : i + training_step_size]
				tmp_test  = corpus_testing[i : i + training_step_size]

				# produce X and y
				X_train, y_train = sample_sents(tmp_train, C)
				X_test,  y_test  = sample_sents(tmp_test,  C)

				model.fit(X_train, y_train, verbose = 0)
				#model.fit_generator(sample_word(corpus_subsample, vocab), samples_per_epoch=10000, nb_epoch=10)
				accuracy = model.evaluate(X_test, y_test)
				accuracy = accuracy[1]
				print "Accuracy: ", accuracy, "%"

				iteration += 1


	def predict(word):
		"""After the model is trained, use the trained model to predict the given word's
		   word vector. This is done by looking up the weights in the Keras model.
		"""

		try:
			return self.model.layers[0].get_weights()[0][vocab.index(word),]
		except KeyError:
			print "The word is not in the vocabulary"
			return None


