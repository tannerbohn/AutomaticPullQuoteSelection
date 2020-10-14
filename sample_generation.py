import numpy as np
import time
import random
import itertools
import argparse
import pickle
import os
import textwrap

from sklearn.linear_model import LogisticRegression

import settings
from utils.pq_preprocessing import preprocess_pull_quotes
from utils.data_utils import get_article_partition
from utils.Evaluator import Evaluator
from utils.misc_utils import get_sentences

from models.SimplePQModel import SimplePQModel


def analyze_text(model, text):
	sentences = get_sentences(text)
	scores = model.predict_article(sentences, sentences)
	scores = zip(range(len(sentences)), sentences, scores)
	scores = sorted(scores, key = lambda el: -el[2])
	for i in range(min(len(sentences), 10)):
		index, sentence, score = scores[i]
		print("{}\t{:.2f}\t{}\n".format(index, 100*score, sentence.replace('"', "“")))

def get_pq_source_sents(articles):

	results = []

	for article in articles:
		r = []
		for i_pq, pq in enumerate(article['edited_pqs']):
			source_sentences = [s for s, inclusion in zip(article['sentences'], article['inclusions']) if inclusion == i_pq+1]
			r.append(' '.join(source_sentences))
		results.append(r)

	return results


def wrap_line(text, width, add_indent=True):

	'''
	new_lines = []
	i = 0
	while len(text) > 0:
		new_lines.append("{}{}".format("\t" if add_indent and i > 0 else "", text[:width]))
		text = text[width:]
		if len(text) < 10 and len(text) > 0:
			new_lines[-1] += text
			text = ""
		i += 1
	'''
	new_lines = textwrap.wrap(text, width=width)

	if add_indent:
		return "\n\t____".join(new_lines)
	else:
		return '\n'.join(new_lines)



def generate_samples(model, articles):
	# url
	# true PQs with their source texts
	# top 3 sentences with probabilities
	top_sents = []

	for i_article, article in enumerate(articles):

		scores = model.predict_article(article['sentences'], article['sentences'])
		scores = zip(article['sentences'], scores)
		scores = sorted(scores, key = lambda el: -el[1])

		top_sent = scores[0][0]
		top_sents.append(top_sent.replace('"', "“"))

		#for i_pq, pq in enumerate(article['edited_pqs']):
		#	source_sentences = [s for s, inclusion in zip(article['sentences'], article['inclusions']) if inclusion == i_pq+1]
	return top_sents

def convert_combined_dict_to_txt(combined_samples, filename):
	f = open(filename, "w")
	nb_articles = len(combined_samples['urls'])
	for i_article in range(nb_articles):
		f.write("Sample {} ({})\n".format(i_article, combined_samples['urls'][i_article]))
		f.write("Model\tHighest rated sentence(s)\n")
		for i_pq, pq in enumerate(combined_samples['PQ Sources'][i_article]):
			if i_pq == 0:
				f.write("True PQ Source\t{}\n".format(wrap_line(pq, width=120)))
			else:
				f.write("\t{}\n".format(wrap_line(pq, width=120)))
		keys = list(combined_samples.keys())
		keys.remove('urls')
		keys.remove("PQ Sources")
		for k in keys:
			f.write("{}\t{}\n".format(k, wrap_line(combined_samples[k][i_article], width=120)))
		f.write("\n")
	f.close()

parser = argparse.ArgumentParser(description='Specify model name')
parser.add_argument('model_name')
parser.add_argument('--quick', action="store_true", default=False)
parsing = parser.parse_args()
model_name = parsing.model_name
quick_mode = parsing.quick
print("MODEL NAME:", model_name)
print("QUICK MODE:", quick_mode)

#_ = input("?")

assert model_name in ["handcrafted", "ngrams", "c_deep", "clickbait", "headline", "summarizers"]


articles_data = preprocess_pull_quotes(directories=settings.PQ_SAMPLES_DIRS)
if quick_mode: articles_data = articles_data[:100]
#print("# articles = ", len(articles_data))

train_articles, val_articles, test_articles = get_article_partition(articles_data, train_frac=0.7, val_frac=0.1, test_frac=0.2, seed=1337, verbose=1)


combined_samples_fname = "results/pq_samples/combined_samples.pkl"
if os.path.exists(combined_samples_fname):
	with open(combined_samples_fname, "rb") as f:
		combined_samples = pickle.load(f)
else:
	combined_samples = dict()
	combined_samples['urls'] = [a['url'] for a in test_articles]
	combined_samples['PQ Sources'] = get_pq_source_sents(test_articles)

###############################################################################

if model_name == "handcrafted":
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.tree import DecisionTreeClassifier
	from models.sentence_encoders import HandcraftedEncoder



	#sent_encoder = HandcraftedEncoder()
	sent_encoder = HandcraftedEncoder(precomputed_embeddings=settings.PRECOMPUTED_HANDCRAFTED_EMBEDDINGS_FNAME)
	feature_list = ["Quote_count", "Sent_position", "R_difficult", "POS_PRP", "POS_VB", "A_concreteness"] #HandcraftedEncoder._all_features + "best"
	#feature = "best"


	for feature in feature_list:
		print(feature)
		sent_encoder.set_features(feature)
		model = SimplePQModel(sent_encoder=sent_encoder, clf_type=AdaBoostClassifier, clf_args={'n_estimators':100, 'base_estimator':DecisionTreeClassifier(max_depth=1, class_weight="balanced")})
		print("training {}...".format(feature))
		model.fit(train_articles)
		print("generating...")

		combined_samples[feature] = generate_samples(model, test_articles)



elif model_name == "ngrams":
	from models.sentence_encoders import NGramEncoder

	for mode, n in [('char', 2), ('word', 1)]:
		print(mode, n)
		sent_encoder = NGramEncoder(mode=mode, n=n, store_results=False, vocab_size=1000)
		print("preparing encoder...")
		sent_encoder.fit(train_articles)
		print("done")
		model = SimplePQModel(sent_encoder=sent_encoder, clf_type=LogisticRegression, clf_args={'class_weight':'balanced', 'max_iter':1000})
		model.fit(train_articles)
		
		combined_samples["{}-{}".format(mode.capitalize(), n)] = generate_samples(model, test_articles)




elif model_name == "c_deep":
	from models.sentence_encoders import SentBERTEncoder
	from models.FlexiblePQModel import FlexiblePQModel

	sent_encoder = SentBERTEncoder(precomputed_embeddings=settings.PRECOMPUTED_SBERT_EMBEDDINGS_FNAME)

	model = FlexiblePQModel(sent_encoder=sent_encoder, mode="C_deep")
	model.prepare_data(train_articles, val_articles)
	model.train_model(nb_experts=4, width=32)

	combined_samples["C_deep"] = generate_samples(model, test_articles)





elif model_name == "headline":
	from models.sentence_encoders import SentBERTEncoder
	from models.HeadlinePopularityPQModel import HeadlinePopularityPQModel

	sent_encoder = SentBERTEncoder(precomputed_embeddings=settings.PRECOMPUTED_SBERT_EMBEDDINGS_FNAME)

	model = HeadlinePopularityPQModel(sent_encoder=sent_encoder, dataset_fname=settings.HEADLINE_POPULARITY_FNAME, quick_mode=quick_mode)
	model.fit(train_articles)
	combined_samples["Headline popularity"] = generate_samples(model, test_articles)


elif model_name == "clickbait":
	from models.sentence_encoders import SentBERTEncoder
	from models.ClickbaitPQModel import ClickbaitPQModel

	sent_encoder = SentBERTEncoder(precomputed_embeddings=settings.PRECOMPUTED_SBERT_EMBEDDINGS_FNAME)

	model = ClickbaitPQModel(sent_encoder=sent_encoder, clickbait_fname=settings.CLICKBAIT_FNAME, headlines_fname=settings.HEADLINE_FNAME, quick_mode=quick_mode)
	model.fit(train_articles)
	combined_samples["Clickbait"] = generate_samples(model, test_articles)





elif model_name == "summarizers":
	from models.SummarizerPQModel import SummarizerPQModel

	for name in ["TextRankSummarizer"]: #["LexRankSummarizer", "SumBasicSummarizer", "KLSummarizer", "TextRankSummarizer"]:
		print(name)
		model = SummarizerPQModel(name=name)
		combined_samples["{}".format(name)] = generate_samples(model, test_articles)


print("Saving samples...")
with open(combined_samples_fname, "wb") as f:
	pickle.dump(combined_samples, f)

print("Done")

convert_combined_dict_to_txt(combined_samples, "results/pq_samples/combined_samples.txt")