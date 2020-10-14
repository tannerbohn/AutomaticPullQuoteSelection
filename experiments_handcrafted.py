import time
import argparse

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from models.sentence_encoders import HandcraftedEncoder
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
if quick_mode: articles_data = articles_data[:1000]

train_articles, val_articles, test_articles = get_article_partition(articles_data, train_frac=0.7, val_frac=0.1, test_frac=0.2, seed=1337, verbose=1)

E = Evaluator()


if settings.PRECOMPUTED_HANDCRAFTED_EMBEDDINGS_FNAME != None:
	sent_encoder = HandcraftedEncoder(precomputed_embeddings=settings.PRECOMPUTED_HANDCRAFTED_EMBEDDINGS_FNAME)
else:
	sent_encoder = HandcraftedEncoder()
	print("precomputing sentence features...")
	sent_encoder.precompute(train_articles+val_articles+test_articles)
	print("done")



timestamp = time.strftime("%F-%H:%M:%S")
results_file = open("results/handcrafted_{}.txt".format(timestamp), "w")

feature_list = HandcraftedEncoder._all_features


chosen_features = []

max_val_score = 0

while len(chosen_features) < 1:
	best_val = 0.
	best_feature = None

	for feature in feature_list:
		if feature in chosen_features: continue
	
		sent_encoder.feature_names = chosen_features+[feature]
		model = SimplePQModel(sent_encoder=sent_encoder, clf_type=AdaBoostClassifier, clf_args={'n_estimators':100, 'base_estimator':DecisionTreeClassifier(max_depth=1, class_weight="balanced")})
		model.fit(train_articles)

		val_accuracy = E.evaluate(model=model, articles=val_articles, verbose=0)
		test_accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)
		
		res_str = "{}\t{:.1f}\t{:.1f}".format(', '.join(chosen_features+[feature]), 100*val_accuracy, 100*test_accuracy)
		print(res_str)
		results_file.write(res_str+"\n")
		results_file.flush()
		
		if val_accuracy > best_val:
			best_val = val_accuracy
			best_feature = feature
	
	print("\n")
	results_file.write("\n")

	if best_val > max_val_score:
		max_val_score = best_val
		chosen_features.append(best_feature)
	else:
		break

	if len(chosen_features) >= len(feature_list):
		break

results_file.close()