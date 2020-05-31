
import sys, os
CODE_FILE_LOC = "/".join(os.path.realpath(__file__).split("/")[:-1])
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, CODE_FILE_LOC)




import numpy as np
import pickle
from collections import Counter

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
		"R_CLI",
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

	_best_features = [
		"Quote_count",
		"Sent_position",
		"A_concreteness",
		"Len_word_avg",
		"A_neg",
		"POS_MD",
		"POS_PRP"
	]


	def __init__(self, features="all"):

		self.name = "enc_handcrafted"

		if features == "all":
			self.feature_names = self._all_features
		elif features == "best":
			self.feature_names = self._best_features
		else:
			self.feature_names = features

		self._precomputed = dict()
		for name in self._all_features:
			self._precomputed[name] = dict()

		return

	def reset(self):
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

		for i_a, a in enumerate(articles):
			if i_a % 100 == 0: print("{}%".format(100*i_a/len(articles)))
			for s in a['sentences']:
				#for feature in self._all_features:
				#	_ = self.precompute_feature_value(feature, s, a[sentences])
				_ = self.encode(s, a['sentences'])

		self.feature_names = [n for n in old_features]

		return

	def encode(self, text, document=None):

		doc_start = "" if document == None else document[0][:20]

		vec = []

		for feature in self.feature_names:

			try:
				val = self._precomputed[feature][text+doc_start]
				vec.append(val)
			except:
				val = self.precompute_feature_value(feature, text, document)
				vec.append(val)

		return vec

	def precompute_feature_value(self, feature, text, document):

		sentence = text
		doc_start = "" if document == None else document[0][:20]


		if feature in ["A_valence", "A_arousal", "A_concreteness"]:
			valence_avg, arousal_avg, concreteness_avg = hf.get_vac(sentence)
			self._precomputed["A_valence"][text+doc_start] = valence_avg
			self._precomputed["A_arousal"][text+doc_start] = arousal_avg
			self._precomputed["A_concreteness"][text+doc_start] = concreteness_avg

		
		if feature in ["A_pos", "A_neg", "A_compound"]:
			pos_sentiment, neg_sentiment, compound_sentiment = hf.get_sentiment_values(sentence)
			self._precomputed["A_pos"][text+doc_start] = pos_sentiment
			self._precomputed["A_neg"][text+doc_start] = neg_sentiment
			self._precomputed["A_compound"][text+doc_start] = compound_sentiment


		if feature == "Quote_count":
			self._precomputed[feature][text+doc_start] = hf.quote_presence(sentence)
		elif feature == "Sent_position":
			self._precomputed[feature][text+doc_start] = hf.get_sentence_location(sentence, document)
		elif feature == "Len_total":
			self._precomputed[feature][text+doc_start] = hf.get_total_length(sentence)
		

		elif feature == "R_Flesch" in self.feature_names:
			self._precomputed[feature][text+doc_start] = hf.get_readability_score(sentence, metric="flesch")
		elif feature == "R_CLI" in self.feature_names:
			self._precomputed[feature][text+doc_start] = hf.get_readability_score(sentence, metric="coleman_liau_index")
		elif feature == "R_difficult" in self.feature_names:
			self._precomputed[feature][text+doc_start] = hf.get_readability_score(sentence, metric="difficult_words")
		elif feature == "Len_word_avg":
			self._precomputed[feature][text+doc_start] = hf.get_readability_score(sentence, metric="avg_word_length")

		elif feature.startswith("POS_"):
			tags = feature[4:].split("+")
			self._precomputed[feature][text+doc_start] = hf.get_pos_tag_score(sentence, tags)

		return self._precomputed[feature][text+doc_start]


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

		self.sent_encoder = SentenceTransformer("bert-base-nli-mean-tokens")

		self._precomputed_sent_encs = dict()

		if precomputed_embeddings != None:
			print("loading precomputed embeddings...")
			self.load_precomputed_embeddings(precomputed_embeddings)
			print("done")

		return

	def load_precomputed_embeddings(self, filename):

		with open(filename, "rb") as f:
			self._precomputed_sent_encs = pickle.load(f)

	def reset(self):
		self._precomputed_sent_encs = dict()

	def precompute(self, sentences):

		vecs = self.sent_encoder.encode(sentences, show_progress_bar=True)
		for i_s, sentence in enumerate(sentences):
			self._precomputed_sent_encs[sentence] = vecs[i_s]


	def encode(self, text, document=None):

		if type(text) == list:
			encs = [self.encode(sentence) for sentence in text]
			return np.average(encs, axis=0)

		sentence = text


		try:
			return self._precomputed_sent_encs[sentence]
		except:

			vec = self.sent_encoder.encode([sentence], show_progress_bar=False)[0]
			self._precomputed_sent_encs[sentence] = vec

			return vec

	def batch_encode(self, texts, verbose=0):

		vecs = self.sent_encoder.encode(texts, show_progress_bar=verbose>0)
		return vecs



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

		for s in sentences:

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


def get_char_gram_seq(text, n, lower=True):
	#text = ' '.join(split_punct(text))
	if lower:
		text = text.lower()
	ngrams = [text[i:i+n] for i in range(len(text)-n)]
	return ngrams

def get_word_gram_seq(text, n, remove_stops, tokenizer_mode=0):
	words = get_words(text, remove_stops=remove_stops, tokenizer_mode=tokenizer_mode)
	ngrams = [tuple(words[i:i+n]) for i in range(max(0, len(words)-n))]
	return ngrams

def get_pos_gram_seq(text, n):
	tags = hf.get_pos_tags(text, only_tags=True)
	ngrams = [tuple(tags[i:i+n]) for i in range(max(0, len(tags)-n))]
	return ngrams



def get_words(text, remove_stops=True, tokenizer_mode=0):
	
	words = split_punct(text.lower())
	if remove_stops:
		words = [w for w in words if w not in hf.STOPS]
	return words
			
