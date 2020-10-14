
import sys, os
CODE_FILE_LOC = "/".join(os.path.realpath(__file__).split("/")[:-1])
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, CODE_FILE_LOC)




import numpy as np
import pickle
from collections import Counter
from tqdm import tqdm

from utils import misc_utils
from utils.pq_preprocessing import split_punct

import settings

import handcrafted_features as hf

class HandcraftedEncoder:

	_all_features = [
		"Len_total",
		"Sent_position",
		"Quote_count",
		
		"R_Flesch",
		"R_difficult",
		"Len_word_avg",

		
		"POS_CD",
		"POS_JJ",
		"POS_MD",
		"POS_NN", 
		"POS_NNP", 
		"POS_PRP", 
		"POS_RB", 
		"POS_VB",

		"A_pos",
		"A_neg",
		"A_compound",

		"A_valence",
		"A_arousal",
		"A_concreteness"

	]

	# Quote_count, A_concreteness, Sent_position, POS_PRP, A_neg, POS_VB, A_arousal, POS_NNP, POS_CD, R_difficult, POS_JJ, A_valence
	_best_features = [
		"Quote_count",
		"A_concreteness",
		"Sent_position",
		"POS_PRP",
		"A_neg",
		"POS_VB",
		"A_arousal",
		"POS_NNP",
		"POS_CD",
		"R_difficult",
		"POS_JJ",
		"A_valence"		
	]


	def __init__(self, features="all", precomputed_embeddings=None):

		self.name = "enc_handcrafted"

		if features == "all":
			self.feature_names = self._all_features
		elif features == "best":
			self.feature_names = self._best_features
		else:
			self.feature_names = features


		if precomputed_embeddings != None:
			print("loading precomputed handcrafted embeddings...")
			self.load_precomputed_embeddings(precomputed_embeddings)
			print("done")
		else:
			self._precomputed = dict()
			# store a mapping from text to index key
			self._precomputed['text_indices'] = dict()
			self._max_text_index_key = 0
			# for each feature, have a mapping from index key to feature value
			for name in self._all_features:
				self._precomputed[name] = dict()

		return

	def load_precomputed_embeddings(self, filename):
		# embedding should be saved with following command:
		# with open(filename, "wb") as f:
		#	pickle.dump(H._precomputed, f)

		with open(filename, "rb") as f:
			self._precomputed = pickle.load(f)

		self._max_text_index_key = max(self._precomputed['text_indices'].values())

	def reset(self):
		del self._precomputed
		self._precomputed = dict()
		self._precomputed['text_indices'] = dict()
		self._max_text_index_key = 0
		for name in self._all_features:
			self._precomputed[name] = dict()
		return

	def set_features(self, features):
		if features == "best":
			self.feature_names = self._best_features
		elif features == "all":
			self.feature_names = self._all_features
		elif type(features) == str:
			self.feature_names = [features]
		elif type(features) == list:
			self.feature_names = features

	def precompute(self, articles):
		# not yet supported

		old_features = [n for n in self.feature_names]

		self.feature_names = self._all_features

		for i_a, a in enumerate(tqdm(articles)):
			#if i_a % 100 == 0: print("{:.2f}%".format(100*i_a/len(articles)))
			for s in a['sentences']:

				text_key = s+a['sentences'][0][:20]

				self._precomputed['text_indices'][text_key] = self._max_text_index_key + 1
				self._max_text_index_key += 1
				#for feature in self._all_features:
				#	_ = self.precompute_feature_value(feature, s, a[sentences])
				_ = self.encode(s, a['sentences'])

		self.feature_names = [n for n in old_features]

		return

	def encode(self, text, document=None):

		doc_start = "" if document == None else document[0][:20]

		text_key = text+doc_start		


		vec = []

		for feature in self.feature_names:

			try:
				text_index_key = self._precomputed["text_indices"][text_key]
				val = self._precomputed[feature][text_index_key]
				vec.append(val)
			except:
				#print("missed", document==None)
				val = self.precompute_feature_value(feature, text, document)
				vec.append(val)

		return vec

	def precompute_feature_value(self, feature, text, document):

		sentence = text
		doc_start = "" if document == None else document[0][:20]

		text_key = text+doc_start

		try:
			text_index_key = self._precomputed['text_indices'][text_key]
		except:
			text_index_key = self._max_text_index_key + 1
			self._precomputed['text_indices'][text_key] = text_index_key
			self._max_text_index_key += 1

		self._precomputed['text_indices'][text_key] = text_index_key


		if feature in ["A_valence", "A_arousal", "A_concreteness"]:
			valence_avg, arousal_avg, concreteness_avg = hf.get_vac(sentence)
			self._precomputed["A_valence"][text_index_key] = valence_avg
			self._precomputed["A_arousal"][text_index_key] = arousal_avg
			self._precomputed["A_concreteness"][text_index_key] = concreteness_avg

		
		if feature in ["A_pos", "A_neg", "A_compound"]:
			pos_sentiment, neg_sentiment, compound_sentiment = hf.get_sentiment_values(sentence)
			self._precomputed["A_pos"][text_index_key] = pos_sentiment
			self._precomputed["A_neg"][text_index_key] = neg_sentiment
			self._precomputed["A_compound"][text_index_key] = compound_sentiment


		if feature == "Quote_count":
			self._precomputed[feature][text_index_key] = hf.quote_presence(sentence)
		elif feature == "Sent_position":
			self._precomputed[feature][text_index_key] = hf.get_sentence_location(sentence, document)
		elif feature == "Len_total":
			self._precomputed[feature][text_index_key] = hf.get_total_length(sentence)
		

		elif feature == "R_Flesch" in self.feature_names:
			self._precomputed[feature][text_index_key] = hf.get_readability_score(sentence, metric="flesch")
		elif feature == "R_CLI" in self.feature_names:
			self._precomputed[feature][text_index_key] = hf.get_readability_score(sentence, metric="coleman_liau_index")
		elif feature == "R_difficult" in self.feature_names:
			self._precomputed[feature][text_index_key] = hf.get_readability_score(sentence, metric="difficult_words")
		elif feature == "Len_word_avg":
			self._precomputed[feature][text_index_key] = hf.get_readability_score(sentence, metric="avg_word_length")

		elif feature.startswith("POS_"):
			tags = feature[4:].split("+")
			self._precomputed[feature][text_index_key] = hf.get_pos_tag_score(sentence, tags)

		return self._precomputed[feature][text_index_key]

	def get_unique_doc_id(self, document):

		return int(sum([len(s) for s in document]))


	def encode_orig(self, text, document=None):
		if type(text) == list:
			encs = [self.encode(sentence, document) for sentence in text]
			return np.average(encs, axis=0)

		sentence = text

		try:
			vec = self._precomputed_sent_encs[sentence]
			return vec
		except:
		
			if "A_valence" in self.feature_names or "A_arousal" in self.feature_names or "A_concreteness" in self.feature_names:
				valence_avg, arousal_avg, concreteness_avg = hf.get_vac(sentence)
			
			if "A_pos" in self.feature_names or "A_neg" in self.feature_names or "A_compound" in self.feature_names:
				pos_sentiment, neg_sentiment, compound_sentiment = hf.get_sentiment_values(sentence)

			vec = []

			for feature in self.feature_names:

				if feature == "Quote_count":
					vec.append(hf.quote_presence(sentence))
				elif feature == "Sent_position":
					vec.append(hf.get_sentence_location(sentence, document))
				elif feature == "Len_total":
					vec.append(hf.get_total_length(sentence))
				

				elif feature == "R_Flesch" in self.feature_names:
					vec.append(hf.get_readability_score(sentence, metric="flesch"))
				elif feature == "R_CLI" in self.feature_names:
					vec.append(hf.get_readability_score(sentence, metric="coleman_liau_index"))
				elif feature == "R_difficult" in self.feature_names:
					vec.append(hf.get_readability_score(sentence, metric="difficult_words"))
				elif feature == "Len_word_avg":
					vec.append(hf.get_readability_score(sentence, metric="avg_word_length"))

				elif feature.startswith("POS_"):
					tags = feature[4:].split("+")
					vec.append(hf.get_pos_tag_score(sentence, tags))
				
				elif feature == "A_pos":
					vec.append(pos_sentiment)
				elif feature == "A_neg":
					vec.append(neg_sentiment)
				elif feature == "A_compound" in self.feature_names:
					vec.append(compound_sentiment)

				
				elif feature == "A_valence" in self.feature_names:
					vec.append(valence_avg)
				elif feature == "A_arousal" in self.feature_names:
					vec.append(arousal_avg)
				elif feature == "A_concreteness" in self.feature_names:
					vec.append(concreteness_avg)

			self._precomputed_sent_encs[sentence] = vec

			return vec

	'''
	def get_single_feature(self, sentence, feature_name):

		value = 0

		if feature_name in ("A_valence", "A_arousal", "A_concreteness"):
			valence_avg, arousal_avg, concreteness_avg = hf.get_vac(sentence)
		
		if feature_name in ("A_pos", "A_neg", "A_compound"):
			pos_sentiment, neg_sentiment, compound_sentiment = hf.get_sentiment_values(sentence)

		

		if feature_name == "Len_total":
			value = hf.get_total_length(sentence)
		elif feature_name == "Len_total_inv" in self.feature_names:
			value = hf.get_inv_total_length(sentence)
		elif feature_name == "Len_word_avg":
			value = hf.get_avg_word_length(sentence)
		elif feature_name == "Repetition" in self.feature_names:
			value = hf.get_repetition_score(sentence)
		elif feature_name == "R_Flesch" in self.feature_names:
			value = hf.get_readability_score(sentence, metric="flesch")
		elif feature_name == "R_CLI" in self.feature_names:
			value = hf.get_readability_score(sentence, metric="coleman_liau_index")
		elif feature_name == "R_difficult" in self.feature_names:
			value = hf.get_readability_score(sentence, metric="difficult_words")

		elif feature_name.startswith("POS_"):
			tags = feature_name[4:].split("+")
			value = hf.get_pos_tag_score(sentence, tags)
		
		elif feature_name == "A_pos":
			value = pos_sentiment
		elif feature_name == "A_neg":
			value = neg_sentiment
		elif feature_name == "A_compound" in self.feature_names:
			value = compound_sentiment
		elif feature_name == "A_contrast" in self.feature_names:
			value = hf.get_ganguly_sentiment_score(sentence)
		
		elif feature_name == "A_valence" in self.feature_names:
			value = valence_avg

		elif feature_name == "A_arousal" in self.feature_names:
			value = arousal_avg

		#elif feature == "A_concreteness" in self.feature_names:
		#	vec.append(concreteness_avg)

		elif feature_name == "test":
			value = hf.get_experimental_feature(sentence)

		return value
	'''



class FastTextEncoder:


	def __init__(self):
		

		self.name = "fasttext"

		self._precomputed_sent_encs = dict()

		self.WORD_VECS = misc_utils.load_word_vecs(settings.FASTTEXT_FNAME)

		self._zeros = np.zeros(len(self.WORD_VECS['cat']))

	def reset(self):
		self._precomputed_sent_encs = dict()

	def precompute(self, sentences):
		# not yet supported
		return

	def encode(self, text, document=None):

		if type(text) == list:
			encs = [self.encode(sentence) for sentence in text]
			return np.average(encs, axis=0)

		sentence = text

		try:
			return self._precomputed_sent_encs[sentence]
		except:


			words = split_punct(sentence)
			words = [w for w in words if w not in misc_utils.PUNCT]

			vecs = []
			for w in words:
				try:
					vecs.append(self.WORD_VECS[w])
				except:
					pass

			if len(vecs) == 0:
				return self._zeros

			enc = np.average(vecs, axis=0)

			self._precomputed_sent_encs[sentence] = enc
			return enc







class SentBERTEncoder:

	def __init__(self, precomputed_embeddings=None):

		

		# https://arxiv.org/pdf/1908.10084.pdf
		# https://github.com/UKPLab/sentence-transformers

		'''
		"roberta-base-nli-stsb-mean-tokens" - 768 dim
		"bert-base-nli-stsb-mean-tokens" - 768
		"bert-base-nli-max-tokens" 
		"bert-base-nli-mean-tokens"
		'''
		from sentence_transformers import SentenceTransformer

		self.name = "sentbert"

		self.shape = (768,)

		self.sent_encoder = SentenceTransformer("bert-base-nli-mean-tokens")

		self._precomputed_sent_encs = dict()

		# be able to use a subset of the embedding dimensions
		self.dimensions = "all"

		if precomputed_embeddings != None:
			print("loading precomputed embeddings...")
			self.load_precomputed_embeddings(precomputed_embeddings)
			print("done")

		return

	def load_precomputed_embeddings(self, filename):

		with open(filename, "rb") as f:
			self._precomputed_sent_encs = pickle.load(f)

	def set_dimensions(self, new_dimensions):
		if type(new_dimensions) == int:
			self.dimensions = (new_dimensions,)
		elif type(new_dimensions) == list:
			self.dimensions = tuple(new_dimensions)
		elif new_dimensions == "all":
			self.dimensions = "all"


	def reset(self):
		self._precomputed_sent_encs = dict()

	def precompute(self, sentences):

		vecs = self.sent_encoder.encode(sentences, show_progress_bar=True)
		for i_s, sentence in enumerate(sentences):
			self._precomputed_sent_encs[sentence] = vecs[i_s]


	def encode(self, text, document=None):

		#if type(text) == list:
		#	encs = [self.encode(sentence) for sentence in text]
		#	return np.average(encs, axis=0)

		sentence = text


		try:
			enc = self._precomputed_sent_encs[sentence]

			if self.dimensions == "all":
				return enc
			elif len(self.dimensions) == 1:
				return np.array([enc[self.dimensions]])
			else:
				return enc[self.dimensions]
		except:

			enc = self.sent_encoder.encode([sentence], show_progress_bar=False)[0]
			self._precomputed_sent_encs[sentence] = enc

			if self.dimensions == "all":
				return enc
			elif len(self.dimensions) == 1:
				return np.array([enc[self.dimensions]])
			else:
				return enc[self.dimensions]

	def batch_encode(self, texts, verbose=0):

		vecs = self.sent_encoder.encode(texts, show_progress_bar=verbose>0)
		if self.dimensions == "all":
			return vecs
		else:
			return vecs[:,self.dimensions]



class NGramEncoder:

	def __init__(self, mode="char", n = 2, vocab_size=500, remove_stops=False, tokenizer_mode=1, binarize=False, lower=True, store_results=False):

		self.name = "ngram"

		# mode is either character (char), word (word), or POS (pos)
		assert mode in ["char", "word", "pos"], "ERROR: invalid n-gram type"
		self.mode = mode
		self.tokenizer_mode = tokenizer_mode
		self.binarize = binarize
		self.lower = lower

		self.remove_stops = remove_stops

		self.n = n
		self.vocab_size = vocab_size

		self._gram_indices = []

		self.store_results = store_results

		self._precomputed = dict()

		return

	def reset(self):
		self._words = dict()
		self._precomputed = dict()

		return

	def encode_sent(self, sentence):

		if self.store_results:
			try:
				return self._precomputed[sentence]
			except:
				pass

		count_vec = [0 for _ in self._gram_indices]

		if self.mode == "char":
			ngrams = get_char_gram_seq(sentence, self.n, self.lower)
		elif self.mode == "word":
			ngrams = get_word_gram_seq(sentence, self.n, self.remove_stops, self.tokenizer_mode)
		elif self.mode == "pos":
			ngrams = get_pos_gram_seq(sentence, self.n)

		for g in ngrams:
			try:
				index = self._gram_indices[g]
				count_vec[index] += 1
			except:
				pass

		if self.store_results:
			self._precomputed[sentence] = count_vec

		return count_vec

	def encode(self, text, document=None):

		if type(text) == list:
			encs = [self.encode(sentence) for sentence in text]
			return np.sum(encs, axis=0)

		sentence = text

		vec = self.encode_sent(sentence)

		if self.binarize:
			return 1*(np.array(vec) > 0)
		else:
			return np.array(vec)

	def fit(self, articles):

		self.reset()

		sentences = []
		for article in articles:
			sentences.extend(article['sentences'])

		self.identify_top_ngrams(sentences)

		return



	def identify_top_ngrams(self, sentences):

		all_ngrams = Counter()

		for s in tqdm(sentences):

			if self.mode == "char":
				ngrams = get_char_gram_seq(s, self.n, self.lower)
			elif self.mode == "word":
				ngrams = get_word_gram_seq(s, self.n, self.remove_stops, tokenizer_mode=self.tokenizer_mode)
			elif self.mode == "pos":
				ngrams = get_pos_gram_seq(s, self.n)

			all_ngrams.update(ngrams)

		most_common = all_ngrams.most_common(self.vocab_size)
		#print("most common:", most_common)

		ngrams = [g for g, _ in most_common]
		self._gram_indices = dict(zip(ngrams, range(len(ngrams))))


	def save_to_dict(self):

		state = dict()
		state['type'] = type(self)
		state['use_context'] = self.use_context
		state['mode'] = self.mode
		state['tokenizer_mode'] = self.tokenizer_mode
		state['binarize'] = self.binarize
		state['lower'] = self.lower
		state['remove_stops'] = self.remove_stops
		state['n'] = self.n
		state['vocab_size'] = self.vocab_size
		state['_gram_indices'] = self._gram_indices


		return state

	def load_from_dict(self, state):

		self.use_context = state['use_context']
		self.mode = state['mode']
		self.tokenizer_mode = state['tokenizer_mode']
		self.binarize = state['binarize']
		self.lower = state['lower']
		self.remove_stops = state['remove_stops']
		self.n = state['n']
		self.vocab_size = state['vocab_size']
		self._gram_indices = state['_gram_indices']


		return self


class PositionalNGramEncoder:

	def __init__(self, mode="char", quantiles=2, n = 2, vocab_size=500, remove_stops=False, tokenizer_mode=1, binarize=False, lower=True, store_results=False):

		self.name = "positional_ngram"

		# mode is either character (char), word (word), or POS (pos)
		assert mode in ["char", "word", "pos"], "ERROR: invalid n-gram type"
		self.mode = mode
		self.tokenizer_mode = tokenizer_mode
		self.binarize = binarize
		self.lower = lower

		self.remove_stops = remove_stops

		self.quantiles = quantiles
		self.n = n
		self.vocab_size = vocab_size

		self._gram_indices = []

		self.store_results = store_results

		self._precomputed = dict()

		return

	def reset(self):
		self._words = dict()
		self._precomputed = dict()

		return

	def encode_sent(self, sentence):
		'''
		if self.mode == "char":
			sentence = '^'+sentence+'_'
		elif self.mode == "word":
			sentence = '^ '+sentence+' _'
		'''

		if self.store_results:
			try:
				return self._precomputed[sentence]
			except:
				pass

		count_vec = np.zeros((len(self._gram_indices), self.quantiles))	

		#count_vec = [0 for _ in self._gram_indices]

		if self.mode == "char":
			ngrams, quantiles = get_char_gram_seq(sentence, self.n, self.lower, quantiles=self.quantiles)
		elif self.mode == "word":
			ngrams, quantiles = get_word_gram_seq(sentence, self.n, self.remove_stops, self.tokenizer_mode, quantiles=self.quantiles)
		elif self.mode == "pos":
			ngrams, quantiles = get_pos_gram_seq(sentence, self.n, quantiles=self.quantiles)

		for g, q in zip(ngrams, quantiles):
			try:
				col_index = q
				row_index = self._gram_indices[g]
				count_vec[row_index][col_index] += 1
			except:
				pass

		count_vec = count_vec.flatten()

		if self.store_results:
			self._precomputed[sentence] = count_vec

		return count_vec

	def encode(self, text, document=None):

		if type(text) == list:
			encs = [self.encode(sentence) for sentence in text]
			return np.sum(encs, axis=0)

		sentence = text

		vec = self.encode_sent(sentence)

		if self.binarize:
			return 1*(np.array(vec) > 0)
		else:
			return np.array(vec)

	def fit(self, articles):

		self.reset()

		sentences = []
		for article in articles:
			sentences.extend(article['sentences'])

		self.identify_top_ngrams(sentences)

		return



	def identify_top_ngrams(self, sentences):

		all_ngrams = Counter()

		for s in tqdm(sentences):
			

			if self.mode == "char":
				#s = '^'+s+'_'
				ngrams = get_char_gram_seq(s, self.n, self.lower)
			elif self.mode == "word":
				#s = '^ '+s+' _'
				ngrams = get_word_gram_seq(s, self.n, self.remove_stops, tokenizer_mode=self.tokenizer_mode)
			elif self.mode == "pos":
				ngrams = get_pos_gram_seq(s, self.n)

			all_ngrams.update(ngrams)

		most_common = all_ngrams.most_common(self.vocab_size)
		#print("most common:", most_common)

		ngrams = [g for g, _ in most_common]
		self._gram_indices = dict(zip(ngrams, range(len(ngrams))))


	def save_to_dict(self):

		state = dict()
		state['type'] = type(self)
		state['use_context'] = self.use_context
		state['mode'] = self.mode
		state['tokenizer_mode'] = self.tokenizer_mode
		state['binarize'] = self.binarize
		state['lower'] = self.lower
		state['remove_stops'] = self.remove_stops
		state['n'] = self.n
		state['quantiles'] = self.quantiles
		state['vocab_size'] = self.vocab_size
		state['_gram_indices'] = self._gram_indices


		return state

	def load_from_dict(self, state):

		self.use_context = state['use_context']
		self.mode = state['mode']
		self.tokenizer_mode = state['tokenizer_mode']
		self.binarize = state['binarize']
		self.lower = state['lower']
		self.remove_stops = state['remove_stops']
		self.n = state['n']
		self.quantiles = state['quantiles']
		self.vocab_size = state['vocab_size']
		self._gram_indices = state['_gram_indices']


		return self


def get_char_gram_seq(text, n, lower=True, quantiles=None):
	#text = ' '.join(split_punct(text))
	if lower:
		text = text.lower()
	ngrams = [text[i:i+n] for i in range(len(text)-n)]
	if quantiles != None:
		return ngrams, get_quantiles_vec(len(ngrams), quantiles)
	else:
		return ngrams

def get_word_gram_seq(text, n, remove_stops, tokenizer_mode=0, quantiles=None):
	words = get_words(text, remove_stops=remove_stops, tokenizer_mode=tokenizer_mode)
	ngrams = [tuple(words[i:i+n]) for i in range(max(0, len(words)-n))]
	if quantiles != None:
		return ngrams, get_quantiles_vec(len(ngrams), quantiles)
	else:
		return ngrams

def get_pos_gram_seq(text, n, quantiles=None):
	tags = hf.get_pos_tags(text, only_tags=True)
	ngrams = [tuple(tags[i:i+n]) for i in range(max(0, len(tags)-n))]
	if quantiles != None:
		return ngrams, get_quantiles_vec(len(ngrams), quantiles)
	else:
		return ngrams



def get_words(text, remove_stops=True, tokenizer_mode=0):
	
	words = split_punct(text.lower())
	if remove_stops:
		words = [w for w in words if w not in hf.STOPS]
	return words
			
def get_quantiles_vec(n, quantiles):
	return list((np.array(range(n))//(n/quantiles)).astype(np.int))
	'''
	vec = [0 for _ in range(n)]
	for i in range(n):
		true_frac = 0 if n <= 1 else i/(n-1)
		bucket = int(min(math.floor(true_frac * quantiles), quantiles-1))
		vec[i] = bucket
	return vec
	'''