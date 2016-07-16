"""
doc2vec.py
@author: Yusan Lin
@description: This program implements the Paragraph Vector proposed in 
[1] Quoc Le, Tomas Mikolov, Distributed Representations of Sentences and Documents, https://cs.stanford.edu/~quocle/paragraph_vector.pdf
"""

import itertools

path_data = "../data/"
stemmer   = PorterStemmer()
TWOPLACES = decimal.Decimal(10) ** -2


def clean(s):
	s = s.replace("<br />", "") # remove all the new line tags
	s = s.replace("\\", "")     # remove all the annoying back slashes
	s = s.lower()               # convert all the letters into lowercases
	s = s.decode("ascii", "ignore")

	s_stemmed = []
	for x in s.split(" "):
		s_stemmed.append(stemmer.stem(x))

	return " ".join(s_stemmed)

print "Processing the corpus data..."
# read in data
filename  = "unlabeledTrainData.tsv"
corpus = [] # initialize corpus into a long long string that concatenates all the reviews together
with open(path_data + filename, "rbU") as f:
	next(f) # skip the first line
	reader = csv.reader(f, delimiter = '\t')
	for row in reader:
		corpus.append(clean(row[1]))

# create vocabularies
# the corpus is flattened into a 1-d list
counter = Counter(chain.from_iterable(corpus).split(" "))
#vocab   = [x for x,y in counter.most_common() if y >= MIN_F]

# get all the unique words and sort them alphabetically
# add in the special token for buffer as well
vocab   = [x for x,y in counter.most_common()]; vocab.append("<s>"); vocab.sort()
N_V     = len(vocab)

# ---------------
# Build the model
# ---------------
N = len(corpus) # number of paragraphs/docs
M = N_V         # number of vocabularies

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

# -----------------
# Sampling the data
# -----------------

