"""
loader.py
"""

from initializer import *
from functions import *

path_data = "data/"

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

def load_corpus_2(filename):
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