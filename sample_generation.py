import numpy as np
import time
import random
import itertools
import argparse

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




def generate_samples(model, articles, filename):
	# url
	# true PQs with their source texts
	# top 3 sentences with probabilities
	results_file = open(filename, "w")
	for i_article, article in enumerate(articles):
		results_file.write("-------------------------------------------------------------------------------\n")
		results_file.write("Test article {} URL: {}\n".format(i_article, article['url']))
		scores = model.predict_article(article['sentences'], article['sentences'])
		scores = zip(range(len(article['sentences'])), article['sentences'], scores)
		scores = sorted(scores, key = lambda el: -el[2])
		for i_pq, pq in enumerate(article['edited_pqs']):
			results_file.write("Edited PQ {}: {}\n".format(i_pq+1, pq))
			source_sentences = [s for s, inclusion in zip(article['sentences'], article['inclusions']) if inclusion == i_pq+1]
			results_file.write("PQ Source {}: {}\n".format(i_pq+1, ' '.join(source_sentences)))
			results_file.write("\n")
		for i in range(3):
			index, sentence, score = scores[i]
			results_file.write("{}\t{:.3f}\t{}\n".format(index, 100*score, sentence.replace('"', "“")))
		results_file.write("\n\n")
	results_file.close()


parser = argparse.ArgumentParser(description='Specify model name')
parser.add_argument('model_name')
parser.add_argument('--quick', action="store_true", default=False)
parsing = parser.parse_args()
model_name = parsing.model_name
quick_mode = parsing.quick
print("MODEL NAME:", model_name)
print("QUICK MODE:", quick_mode)

#_ = input("?")

assert model_name in ["h_combined", "ngrams", "ppd", "sbert", "clickbait", "headline", "summarizers"]


articles_data = preprocess_pull_quotes(directories=settings.PQ_SAMPLES_DIRS)
if quick_mode: articles_data = articles_data[:100]
#print("# articles = ", len(articles_data))

train_articles, val_articles, test_articles = get_article_partition(articles_data, train_frac=0.7, val_frac=0.1, test_frac=0.2, seed=1337, verbose=1)



###############################################################################

if model_name == "h_combined":
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.tree import DecisionTreeClassifier
	from models.sentence_encoders import HandcraftedEncoder

	sent_encoder = HandcraftedEncoder()
	#feature_list = ["Quote_count", "Sent_position", "R_difficult", "POS_PRP", "POS_VB", "A_concreteness", "best"] #HandcraftedEncoder._all_features + "best"
	feature_list = ["best"]

	for feature in feature_list:
		print(feature)
		sent_encoder.set_features(feature)
		model = SimplePQModel(sent_encoder=sent_encoder, clf_type=AdaBoostClassifier, clf_args={'n_estimators':100, 'base_estimator':DecisionTreeClassifier(max_depth=1, class_weight="balanced")})
		print("training {}...".format(feature))
		model.fit(train_articles)
		print("generating...")
		generate_samples(model, test_articles[:100], "results/pq_samples/handcrafted_{}.txt".format(feature))



elif model_name == "ngrams":
	from models.sentence_encoders import NGramEncoder

	for mode, n in [('char', 2), ('word', 1), ('pos', 2)]:
		print(mode, n)
		sent_encoder = NGramEncoder(mode=mode, n=n, store_results=False, vocab_size=1000)
		print("preparing encoder...")
		sent_encoder.fit(train_articles)
		print("done")
		model = SimplePQModel(sent_encoder=sent_encoder, clf_type=LogisticRegression, clf_args={'class_weight':'balanced', 'max_iter':1000})
		model.fit(train_articles)
		generate_samples(model, test_articles[:100], "results/pq_samples/ngram_{}-{}.txt".format(mode, n))





elif model_name == "ppd":
	from models.sentence_encoders import SentBERTEncoder
	from models.PPDEncoder import PPDEncoder

	pre_sent_encoder = SentBERTEncoder(precomputed_embeddings=settings.PRECOMPUTED_SBERT_EMBEDDINGS_FNAME)

	random.shuffle(train_articles)
	sent_encoder = PPDEncoder(sent_encoder=pre_sent_encoder, quantiles=20, layer_sizes=(128, 64), layer_dropouts=(0.25, 0.1), activation='selu')
			
	sent_encoder.fit(train_articles[:int(len(train_articles)*0.75)], val_articles=val_articles, verbose=1)

	model = SimplePQModel(sent_encoder=sent_encoder, clf_type=LogisticRegression, clf_args={'class_weight':'balanced', 'max_iter':1000})
	model.fit(train_articles)

	generate_samples(model, test_articles[:100], "results/pq_samples/ppd_{}.txt".format(20))




elif model_name == "sbert":
	from models.sentence_encoders import SentBERTEncoder

	sent_encoder = SentBERTEncoder(precomputed_embeddings=settings.PRECOMPUTED_SBERT_EMBEDDINGS_FNAME)

	model = SimplePQModel(sent_encoder=sent_encoder, clf_type=LogisticRegression, clf_args={'class_weight':'balanced', 'max_iter':1000})

	model.fit(train_articles)

	generate_samples(model, test_articles[:100], "results/pq_samples/sbert.txt")




elif model_name == "clickbait":
	from models.sentence_encoders import SentBERTEncoder
	from models.ClickbaitPQModel import ClickbaitPQModel

	sent_encoder = SentBERTEncoder(precomputed_embeddings=settings.PRECOMPUTED_SBERT_EMBEDDINGS_FNAME)

	model = ClickbaitPQModel(sent_encoder=sent_encoder, clickbait_fname=settings.CLICKBAIT_FNAME, headlines_fname=settings.HEADLINE_FNAME, quick_mode=quick_mode)
	model.fit(train_articles)
	generate_samples(model, test_articles[:100], "results/pq_samples/clickbait.txt")


elif model_name == "headline":
	from models.sentence_encoders import SentBERTEncoder
	from models.HeadlinePopularityPQModel import HeadlinePopularityPQModel

	sent_encoder = SentBERTEncoder(precomputed_embeddings=settings.PRECOMPUTED_SBERT_EMBEDDINGS_FNAME)

	model = HeadlinePopularityPQModel(sent_encoder=sent_encoder, dataset_fname=settings.HEADLINE_POPULARITY_FNAME, quick_mode=quick_mode)
	model.fit(train_articles)
	generate_samples(model, test_articles[:100], "results/pq_samples/headline_popularity.txt")


elif model_name == "summarizers":
	from models.SummarizerPQModel import SummarizerPQModel

	for name in ["LexRankSummarizer", "SumBasicSummarizer", "KLSummarizer", "TextRankSummarizer"]:
		print(name)
		model = SummarizerPQModel(name=name)
		generate_samples(model, test_articles[:100], "results/pq_samples/summarizer_{}.txt".format(name))
