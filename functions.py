"""
functions.py
"""

from initializer import *

def clean(s):
	s = s.replace("<br />", "") # remove all the new line tags
	s = s.replace("\\", "")     # remove all the annoying back slashes
	s = s.lower()               # convert all the letters into lowercases
	s = s.decode("ascii", "ignore")

	s_stemmed = []
	for x in s.split(" "):
		s_stemmed.append(stemmer.stem(x))

	return " ".join(s_stemmed)