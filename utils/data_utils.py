import sys, os
CODE_FILE_LOC = "/".join(os.path.realpath(__file__).split("/")[:-1])
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, CODE_FILE_LOC)

import random
import numpy as np
import pq_preprocessing
import misc_utils
import settings

PRECOMPUTED_FALSE_INCLUSIONS = dict()


def load_pq_articles(pq_dirs):

	'''
	Option 1: identify a random subset of the sentences VS the real set
	(presumably a model for this can be used to prune down to the sub-sentence level)
	'''

	pull_quotes_data = pq_preprocessing.preprocess_pull_quotes(pq_dirs)

	article_eval_data = []

	for article in pull_quotes_data:

		context = article['body']

		pq_ranking_data = []

		for pq in article['data']:

			X = []
			y = []

			edited_pq_text = pq['pull_quote']
			sentences = pq['sentences']
			true_inclusions = tuple(pq['inclusions'])

			#false_inclusions = generate_random_inclusions(n=nb_subsets-1, l=len(true_inclusions), avoid=true_inclusions)
			false_inclusions = generate_all_inclusions(dim=len(true_inclusions), m=sum(true_inclusions), M=sum(true_inclusions))
			try:
				false_inclusions.remove(true_inclusions)
			except:
				pass
			#print("NB FALSE INCLUSIONS:", len(false_inclusions))

			x_sentences = []
			for s, inc in zip(sentences, true_inclusions):
				if inc == 1:
					x_sentences.append(s)

			#X.append(" ".join(x_sentences))
			X.append((x_sentences, context))
			y.append(1)

			for inclusion_vec in false_inclusions:
				x_sentences = []
				for s, inc in zip(sentences, inclusion_vec):
					if inc == 1:
						x_sentences.append(s)

				#X.append(" ".join(x_sentences))
				X.append((x_sentences, context))
				y.append(0)

			pq_ranking_data.append({"X":X, "y":y, "edited":edited_pq_text})

		article_eval_data.append({"url":article['url'], "all_sentences": pq_ranking_data})


	return article_eval_data




def create_training_data(articles_data, include_edited=False):

	'''
	(at least) two approaches to consider:
		1. aggregate all negative samples into same class, and all pos into same class
		2. perform pairwise transform before aggregating classes

	'''

	X = []
	y = []
	edited_texts = []


	for article in articles_data:

		for pq_data in article['pq_data']:

			POS = []
			NEG = []

			edited = pq_data['edited']

			for (sents, context), label in zip(pq_data['X'], pq_data['y']):
				if include_edited:
					x = (sents, context, edited)
				else:
					x = (sents, context)
				

				if label == 1:
					POS.append(x)
				else:
					NEG.append(x)

			# TODO: implement optional pairwise transform

			for x in POS:
				X.append(x)
				y.append(1)

			for x in NEG:
				X.append(x)
				y.append(0)



	return X, y


def generate_random_inclusions(n, l, avoid):

	assert l > 1, "ERROR: too few sentence to generate non-ground-truth random inclusions"

	assert n < 2**l, "ERROR: too many samples requested for the given inclusion vector length"
	# n: number of random inclusion vectors to make
	# l: the length/dimension of each inclusion vector
	# avoid: the ground truth inclusion vector (do not re-generate)

	avoid_sum = sum(avoid)

	inclusion_vecs = []

	for _ in range(n):

		v = None

		while v == None or v == avoid or v in inclusion_vecs:

			# TODO: does this give a distribution over lengths?
			nb_included = np.random.randint(1, l-1) # always leave at least one left out
			#nb_included = np.random.randint(max(1, avoid_sum-1), avoid_sum+1)

			v = [1 for _ in range(nb_included)]+[0 for _ in range(l - nb_included)]

			np.random.shuffle(v)

		inclusion_vecs.append(v)

	return inclusion_vecs

def get_fold_data(X, i_fold, nb_folds=10):
	n = len(X)
	per_fold = 1.*n/nb_folds
	start_index = int(i_fold*per_fold)
	end_index = int(min(start_index + per_fold, n))
	X_test = X[start_index:end_index]
	X_train = X[:start_index]+X[end_index:]
	return X_train, X_test

def get_article_partition(articles, train_frac=0.8, val_frac=0, test_frac=0.2, seed=123, verbose=1):
	assert train_frac+val_frac+test_frac == 1

	

	if seed != None:
		rand_state = random.getstate()
		np_rand_state = np.random.get_state()

		random.seed(seed)
		np.random.seed(seed)

	sample_indices = list(range(len(articles)))
	random.shuffle(sample_indices)

	nb_train = int(len(articles)*train_frac)
	nb_val = int(len(articles)*val_frac)
	nb_test = len(articles)-nb_train-nb_val
	
	if verbose >= 1:
		print("#Train/#val/#test articles = ", nb_train, nb_val, nb_test)

	train_articles = [articles[i] for i in sample_indices[:nb_train]]
	val_articles = [articles[i] for i in sample_indices[nb_train:nb_train+nb_val]]
	test_articles  = [articles[i] for i in sample_indices[-nb_test:]]

	if seed != None:
		random.setstate(rand_state)
		np.random.set_state(np_rand_state)

	return train_articles, val_articles, test_articles


##############################

def load_pq_simple_data():

	pull_quotes_data = pq_preprocessing.preprocess_pull_quotes(settings.PQ_SAMPLES_DIRS)

	edited_pq_texts = []
	source_pq_texts = []
	context_pq_texts = []
	document_texts = []


	for article in pull_quotes_data:

		data = article['data']

		document_texts.append(article['body'])

		n_samples = len(data)

		for pq in data:

			sentences = pq['sentences']
			true_inclusions = pq['inclusions']


			pq_sentences = []
			non_pq_sentences = []
			for s, inc in zip(sentences, true_inclusions):
				if inc == 1:
					pq_sentences.append(s)
				else:
					non_pq_sentences.append([s])

			#source_pq_texts.append([' '.join(pq_sentences)])
			source_pq_texts.append(pq_sentences)
			edited_pq_texts.append([pq['pull_quote']])
			context_pq_texts.extend(non_pq_sentences)


	return edited_pq_texts, source_pq_texts, context_pq_texts, document_texts



def load_headlines_clickbait(headline_fname, clickbait_fname):

	headline_texts = []
	clickbait_texts = []

	with open(headline_fname, "r") as f:
		lines = f.readlines()

		for l in lines:
			l = l.strip()
			if l == "": continue
			#elif len(l) < 10: print("SHORT HEADLINE:", l)

			#headline_texts.append([l])
			headline_texts.append(misc_utils.get_sentences(l))

	with open(clickbait_fname, "r") as f:
		lines = f.readlines()

		for l in lines:
			l = l.strip()
			if l == "": continue
			#elif len(l) < 10: print("SHORT CLICKBAIT:", l)

			#clickbait_texts.append([l])
			clickbait_texts.append(misc_utils.get_sentences(l))

	return headline_texts, clickbait_texts

def load_famous_quotes(quotes_fname):

	quote_texts = []

	with open(quotes_fname, "r") as f:
		lines = f.readlines()

		for l in lines:
			l = l.strip()
			if l == "": continue
			#elif len(l) < 30: print("SHORT QUOTE:", l)

			#quote_texts.append([l])
			quote_texts.append(misc_utils.get_sentences(l))

	return quote_texts



def generate_all_inclusions(dim, m, M): 
	# dim: the length of the binary vectors
	# m; min #1s
	# M: max #1s

	global PRECOMPUTED_FALSE_INCLUSIONS

	try:
		return PRECOMPUTED_FALSE_INCLUSIONS[(dim, m, M)]
	except:

		all_results = []

		for nb_ones in range(m, M+1):  	
			elem_options = [1 for _ in range(nb_ones)]+[0 for _ in range(dim-nb_ones)]
			all_results.extend(get_all_permutations(elem_options, []))

		PRECOMPUTED_FALSE_INCLUSIONS[(dim, m, M)] = all_results


		return all_results

# The main recursive method 
# to print all possible  
# strings of length k 
def get_all_permutations(elem_options, prefix): 
	  
	# Base case: k is 0, 
	# print prefix 
	if len(elem_options) == 0: 
		return [tuple(prefix)]
	
	all_results = []


	already_considered_new_elem_options = set()

	# One by one add all characters  
	# from set and recursively  
	# call for k equals to k-1 
	for i_el, el in enumerate(elem_options): 

		# Next character of input added 
		newPrefix = prefix + [el]
		new_elems_options = tuple(elem_options[:i_el]+elem_options[i_el+1:])

		if new_elems_options not in already_considered_new_elem_options:
			already_considered_new_elem_options.add(new_elems_options)
			# k is decreased, because  
			# we have added a new character 
			all_results.extend(get_all_permutations(new_elems_options, newPrefix))

	return all_results