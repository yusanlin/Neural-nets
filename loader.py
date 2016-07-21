"""
loader.py
"""

from initializer import *
from functions import *

def load_corpus(filename):
	# read in data
	corpus = [] # initialize corpus into a long long string that concatenates all the reviews together
	with open(path_data + filename, "rbU") as f:
		next(f) # skip the first line
		reader = csv.reader(f, delimiter = '\t')
		for row in reader:
			corpus.append(clean(row[1]))

	# create vocabularies
	# the corpus is flattened into a 1-d list
	counter = Counter(" ".join(corpus).split(" "))
	corpus_flattened = None
	#vocab   = [x for x,y in counter.most_common() if y >= MIN_F]

	# get all the unique words and sort them alphabetically
	# add in the special token for buffer as well
	vocab   = [x for x,y in counter.most_common()]; vocab.append("<s>"); vocab.sort()

	return corpus, vocab