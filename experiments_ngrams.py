import time
import itertools
import argparse

from sklearn.linear_model import LogisticRegression

from models.sentence_encoders import NGramEncoder
from models.SimplePQModel import SimplePQModel

import settings
from utils.pq_preprocessing import preprocess_pull_quotes
from utils.data_utils import get_article_partition
from utils.Evaluator import Evaluator


parser = argparse.ArgumentParser(description='Specify experiment mode')
parser.add_argument('--quick', action="store_true", default=False)
parsing = parser.parse_args()
quick_mode = parsing.quick
if quick_mode:
	print("QUICK MODE")


articles_data = preprocess_pull_quotes(directories=settings.PQ_SAMPLES_DIRS)
if quick_mode: articles_data = articles_data[:100]

train_articles, val_articles, test_articles = get_article_partition(articles_data, train_frac=0.7, val_frac=0.1, test_frac=0.2, seed=1337, verbose=1)

E = Evaluator()


timestamp = time.strftime("%F-%H:%M:%S")
results_file = open("results/ngrams_{}.txt".format(timestamp), "w")

feature_rankings = dict()


for mode, n, vocab_size in itertools.product(["char", "word", "pos"], [1, 2, 3], [1000]):
	sent_encoder = NGramEncoder(mode=mode, n=n, store_results=False, vocab_size=vocab_size)
	print("preparing encoder...")
	sent_encoder.fit(train_articles)
	print("done")
	model = SimplePQModel(sent_encoder=sent_encoder, clf_type=LogisticRegression, clf_args={'class_weight':'balanced', 'max_iter':1000})
	model.fit(train_articles)
	accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)
	res_str = "{}\t{}\t{}\t{}\t{:.1f}".format(sent_encoder.name, mode, n, vocab_size, 100*accuracy)
	print(res_str)

	results_file.write(res_str+"\n")
	results_file.flush()


	# obtain the feature importances and save for printing later
	gram_items = list(model.sent_encoder._gram_indices.items())
	gram_items = sorted(gram_items, key=lambda k: k[1])
	grams = [v[0] for v in gram_items]

	feature_importances = list(zip(grams, model.model.coef_[0]))
	feature_importances = sorted(feature_importances, key=lambda k: k[1])

	print(feature_importances[:20])
	print(feature_importances[-20:])

	feature_rankings["{}-{}-{}".format(mode, n, vocab_size)] = feature_importances



results_file.write("\n\n")

# print highest and lowest weighted features
for key in feature_rankings:
	print(key)
	results_file.write(key+"\n")

	for gram, weight in feature_rankings[key][-20:][::-1]:
		print("{:.3f}\t{}".format(weight, gram))
		results_file.write("{:.3f}\t{}\n".format(weight, gram))
	for gram, weight in feature_rankings[key][:20][::-1]:
		print("{:.3f}\t{}".format(weight, gram))
		results_file.write("{:.3f}\t{}\n".format(weight, gram))
	print()
	results_file.write("\n")
	


results_file.close()
