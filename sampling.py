"""
sampling.py
"""

from initializer import *

def sample_paragraph(corpus, vocab):
	X = []
	Y = []

	N = len(corpus)
	M = len(vocab)

	processed_paragraphs = 0 

	# for the same paragrah, they receive the same paragraph id
	pid = 0 
	for paragraph in corpus:
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
				x = np.zeros(M)

				for j in range(i - C, i + C):
					if i != j:
						tmp = np.zeros(M)
						try:
							tmp[vocab.index(sentence_tmp[j])] = 1
						except ValueError:
							pass
						x += tmp

				x = np.array([ w/float(2 * C) for w in x]).reshape((1, M))[0]

				# the ith word is thus the target word for prediction
				y = np.zeros(M)
				try:
					y[vocab.index(sentence_tmp[i])] = 1
				except ValueError:
					pass
				y = y.reshape((1, M))[0]

				X.append(list(d) + list(x)) # input is the concatenation of paragraph and words
				Y.append(list(y))		

		processed_paragraphs += 1

	X = np.array(X)
	Y = np.array(Y)

	return X, Y

# For the usage of fit_generator()
# this part is the generator that will be fed in later
def sample_word(corpus_subsample, vocab):
	X = []
	Y = []

	N_V = len(vocab)

	# Construct X and Y
	print "Constructing X and Y"
	for sentence in corpus_subsample:

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

			#yield(x, y)

			X.append(list(x))
			Y.append(list(y))

		#processed_sentences += 1

	X = np.array(X)
	Y = np.array(Y)	

	return X, Y