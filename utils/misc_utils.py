
import pickle
import numpy as np
import textblob

PUNCT = "-…/!\"$'(),.:;?@[]‘’“”"

class WordVecHandler:

	def __init__(self, fname):
		print("Loading fasttext embeddings...")
		self.embeddings_dict = pickle.load(open(fname, "rb"), encoding="latin1")
		self._dim = len(self.embeddings_dict['cat'])
		self._zeros = np.zeros(self._dim)
		print("done")

	def __getitem__(self, item):
		#try:
		return self.embeddings_dict[item]
		#except:
		#	#print("not found", item)
		#	return self._zeros

def load_word_vecs(fname):

	word_vecs = WordVecHandler(fname)

	return word_vecs


def load_vac_data(va_fname, c_fname):

	word_data = dict()

	with open(va_fname, "r") as f:

		text = f.readlines()

		col_names = text[0].strip().split(',')

		#print(col_names)

		# want to get values in mean.sum column
		v_col_index = col_names.index('V.Mean.Sum') #V.Mean.Sum
		a_col_index = col_names.index('A.Mean.Sum')
		word_col_index = col_names.index('Word')

		for l in text[1:]:
			if l.strip() == None: break

			col_values = l.strip().split(',')
			valence = float(col_values[v_col_index])
			arousal = float(col_values[a_col_index])
			word = col_values[word_col_index]

			word_data[word] = {"valence":valence, "arousal":arousal}


	with open(c_fname, "r") as f:

		text = f.readlines()

		col_names = text[0].strip().split('\t')

		#print(col_names)

		# want to get values in mean.sum column
		c_col_index = col_names.index('Conc.M')
		word_col_index = col_names.index('Word')

		for l in text[1:]:
			if l.strip() == None: break

			col_values = l.strip().split('\t')
			concreteness = float(col_values[c_col_index])
			word = col_values[word_col_index]

			try:
				word_data[word]["concreteness"] = concreteness
			except:
				word_data[word] = {"concreteness":concreteness}

	#print(word_data["aquatic"])

	return word_data


def separate_words_from_punct(text):
	'''
	! 33 False
	" 34 False
	$ 36 False
	' 39 False
	( 40 False
	) 41 False
	, 44 False
	- 45 False
	. 46 False
	/ 47 False
	: 58 False
	; 59 False
	? 63 False
	@ 64 False
	[ 91 False
	] 93 False
	'''


	puncts = "!\"$'(),-./:;?@[]‘’“”…"

	for ch in puncts:

		text  = text.replace(ch, " "+ch+" ")

	words = text.split()
	text = ' '.join(words)
	return text

def remove_punct(text):

	space_puncts = "-…/"

	no_space_puncts = "!\"$'(),.:;?@[]‘’“”"

	for ch in no_space_puncts:
		text  = text.replace(ch, "")

	for ch in space_puncts:
		text  = text.replace(ch, " ")

	words = text.split()
	text = ' '.join(words)
	return text



def get_sentences(text):
	return [str(s) for s in textblob.TextBlob(text).sentences]


def pretty_num(x, decimals=1):
	if int(x) == x:
		return int(x)
	else:
		if abs(x) > 1:
			return round(x, decimals)
		else:
			s1 = str(x).split('.')[1]
			s2 = s1.lstrip('0')
			nbz = len(s1) - len(s2)
			return "{}.{}{}".format(0, '0'*nbz, s2[:decimals])

def add_ellipses(text, l=100):
	# l: max length of text (including ellipses)

	if len(text) <= l: return text

	else:
		return text[:l-3]+"..."
