"""
initializer.py
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
import itertools

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.engine.topology import *

from collections import Counter
from nltk.stem.porter import PorterStemmer
from sklearn.cross_validation import train_test_split

# constants
N_V       = 1000 # number of vocabularies
N_H       = 50   # number of neurons in the hidden layer
C         = 2    # context window size, meaning including C left words and C right words (2C + 1 in total)
MIN_F     = 100
MAX_ITER  = 10   # maximum allowed training iterations

# common objects
stemmer   = PorterStemmer()
TWOPLACES = decimal.Decimal(10) ** -2