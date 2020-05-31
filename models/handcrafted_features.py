
import numpy as np
import math

from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import wordnet, stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import TweetTokenizer

import textstat
import random
import textblob


from utils import misc_utils
import settings

tknzr = TweetTokenizer()
SID = None
STOPS = set(stopwords.words('english'))
VAC = None # store the contents of valence, arousal, dominance, and concreteness files

with open(settings.EASY_WORDS_FNAME, "r") as f:
	EASY_WORDS = set(f.read().strip().split())


TAGGED = dict()


def quote_presence(text):
	return text.count('“')+text.count('”')

def get_total_length(text):
	# idea length seems to be 20
	#words = text.split()
	return len(text)

def get_sentence_location(text, document):

	index = document.index(text)
	return index/len(document)

def get_readability_score(text, metric="flesch"):
	global tknzr, DIFFICULT

	text = text.replace("’", "'")

	# https://pypi.org/project/textstat/
	if metric == "flesch":
		return textstat.flesch_reading_ease(text)
	elif metric == "smog":
		return textstat.smog_index(text)
	elif metric == "coleman_liau_index":
		return textstat.coleman_liau_index(text)
	elif metric == "automated_readability_index":
		return textstat.automated_readability_index(text)
	elif metric == "dale_chall_readability_score":
		return textstat.dale_chall_readability_score(text)
	elif metric == "difficult_words":
		nb_difficult = 0
		nb_easy = 0
		for w in set(tknzr.tokenize(text.lower())):
			if w not in EASY_WORDS and len(w) >= 6:
				nb_difficult += 1
			else:
				nb_easy += 1
		return 100*nb_difficult/(nb_difficult + nb_easy)
		#return textstat.difficult_words(text)#/len(text.split())
	elif metric == "linsear_write_formula":
		return textstat.linsear_write_formula(text)
	elif metric == "gunning_fog":
		return textstat.gunning_fog(text)
	elif metric == "avg_word_length":
		words = tknzr.tokenize(text)
		words = [w for w in words if w not in misc_utils.PUNCT]
		if len(words) == 0: return 0
		return np.average([len(w) for w in words])

def get_pos_tags(text, only_tags=False):
	global TAGGED

	try:
		tags = TAGGED[text]
	except:
		blob = textblob.TextBlob(text)
		tags = blob.tags
		TAGGED[text] = tags

	if not only_tags:
		return tags
	else:
		return [t[1] for t in tags]

def get_pos_tag_score(text, tag_set):
	global TAGGED

	try:
		tags = TAGGED[text]
	except:
		blob = textblob.TextBlob(text)
		tags = blob.tags
		TAGGED[text] = tags

	nb_words = len(text.split())
	tag_count = 0
	for w, t in tags:
		if t in tag_set:
			tag_count += 1

	density = tag_count/nb_words


	return 100*density

def get_sentiment_values(text):
	global SID

	# https://www.nltk.org/howto/sentiment.html
	if SID == None: SID = SentimentIntensityAnalyzer()

	ss = SID.polarity_scores(text)

	return 100*ss['pos'], 100*ss['neg'], 100*ss['compound']


def get_vac(text):
	global VAC

	# see if we need to load the files
	if VAC == None:  
		VAC = misc_utils.load_vac_data(settings.AFFECT_FNAME, settings.CONCRETENESS_FNAME)

	#words = utils.remove_punct(text).strip().lower().split()

	text = text.replace("’", "'")
	#words = utils.remove_punct(text).split()
	#print("ORIGINAL:", text)
	words = tknzr.tokenize(text.lower())
	#print("BEFORE:", words)
	words = [w for w in words if w not in misc_utils.PUNCT]

	v_default_value = 5
	a_default_value = 4
	c_default_value = 5

	v_scores = []
	a_scores = []
	c_scores = []
	for w in words:
		if w in STOPS: continue
		
		v = get_word_vac_value(w, "valence")
		a = get_word_vac_value(w, "arousal")
		c = get_word_vac_value(w, "concreteness")

		if v != None:
			v_scores.append(v)
		else:
			v_scores.append(v_default_value)

		if a != None:
			a_scores.append(a)
		else:
			a_scores.append(a_default_value)

		if c != None:
			c_scores.append(c)
		else:
			c_scores.append(c_default_value)

	
	if len(v_scores) == 0:
		v_scores = [v_default_value]

	if len(a_scores) == 0:
		a_scores = [a_default_value]

	if len(c_scores) == 0:
		c_scores = [c_default_value]

	v_avg = np.average(v_scores)
	a_avg = np.average(a_scores)

	v_max = np.max(v_scores)
	a_max = np.max(a_scores)


	c_avg = np.average(c_scores)
	#c_max = np.max(c_scores)
	#c_sum = np.sum(c_scores)

	return (v_avg, a_avg, c_avg)

def get_word_vac_value(w, affect):

	worked = False

	value = None

	try:
		value = VAC[w][affect]
	except:
		pass

	return value