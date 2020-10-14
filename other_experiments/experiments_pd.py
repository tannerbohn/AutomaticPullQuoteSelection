import numpy as np
import time
import random
import itertools
import argparse

import keras.backend as K

from sklearn.linear_model import LogisticRegression

from models.sentence_encoders import SentBERTEncoder
from models.SimplePQModel import SimplePQModel
#from models.PDStarMultiEncoder import PDStarMultiEncoder
from models.PDStarEncoder import PDStarEncoder
from models.sentence_encoders import HandcraftedEncoder

import settings

from utils.pq_preprocessing import preprocess_pull_quotes
from utils.data_utils import get_article_partition
from utils.Evaluator import Evaluator


class TruePVDEncoder:

	def __init__(self, feature_calculator, quantiles):
		self.name = "true_pvd_encoder"

		self.feature_calculator = feature_calculator
		self.quantiles = quantiles

	def fit(self, train_articles):

		feature_vals = []
		for a in train_articles:
			feature_vals.extend(self.feature_calculator(a))

		percentile_thresholds = np.percentile(feature_vals, [100*(i)/self.quantiles for i in range(1, self.quantiles)])

		def feature_to_onehot(value):
			onehot_vec = np.zeros(self.quantiles)
			bucket = np.sum(percentile_thresholds < value)
			onehot_vec[bucket] = 1
			return onehot_vec

		self.feature_to_onehot = feature_to_onehot

	def batch_encode(self, document):
		feature_vals = self.feature_calculator(document)

		encs = []

		for v in feature_vals:
			encs.append(self.feature_to_onehot(v))

		return np.array(encs)


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






pre_sent_encoder = SentBERTEncoder(precomputed_embeddings=settings.PRECOMPUTED_SBERT_EMBEDDINGS_FNAME)

h_encoder = HandcraftedEncoder(precomputed_embeddings=settings.PRECOMPUTED_HANDCRAFTED_EMBEDDINGS_FNAME)


#h_encoder.precompute(train_articles+val_articles+test_articles)

#import pickle
#with open("/home/tanner/ml_data/precomputed_handcrafted_embeddings.pkl", "wb") as f:
#	pickle.dump(h_encoder._precomputed, f)

def feature_calculator(article):
	values = []
	sentences = article if type(article) == list else article['sentences']
	for sentence in sentences:
		values.append(h_encoder.encode(sentence, sentences)[0])
	return values

timestamp = time.strftime("%F-%H:%M:%S")
results_file = open("results/pdstar_stage_1_{}.txt".format(timestamp), "w")

quantile_options = [2, 5, 10, 20]


for feature in h_encoder._all_features:
	h_encoder.set_features(feature)
	print("Feature:", feature)
	#

	for quantiles in quantile_options:
		sent_encoder = TruePVDEncoder(feature_calculator=feature_calculator, quantiles=quantiles)
		sent_encoder.fit(train_articles)
		#
		model = SimplePQModel(sent_encoder=sent_encoder, clf_type=LogisticRegression, clf_args={'class_weight':'balanced', 'max_iter':1000, 'solver':'lbfgs'})
		model.fit(train_articles)
		#
		val_accuracy = E.evaluate(model=model, articles=val_articles, verbose=0)
		test_accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)
		# for curiosity, get the weights of the PPD dimensions
		coefs = model.model.coef_[0]
		coef_str = "\t".join([str(round(v, 3)) for v in coefs])
		res_str = "{}\t{}\t{}\t{:.1f}\t{:.1f}\t{:.1f}\t{}".format(sent_encoder.name, feature, quantiles, 100, 100*val_accuracy, 100*test_accuracy, coef_str)
		print(res_str)
		results_file.write(res_str+"\n")
		results_file.flush()

	max_val_acc = 0
	for quantiles in quantile_options:
		sent_encoder = PDStarEncoder(sent_encoder=pre_sent_encoder, feature_calculator=feature_calculator, quantiles=quantiles, layer_sizes=(), layer_dropouts=())
		fith_istory = sent_encoder.fit(train_articles, val_articles=val_articles, verbose=1, max_epochs=1 if quick_mode else 100)
		proxy_val_acc = fit_history['val_acc'][-1] 
		#
		model = SimplePQModel(sent_encoder=sent_encoder, clf_type=LogisticRegression, clf_args={'class_weight':'balanced', 'max_iter':1000, 'solver':'lbfgs'})
		model.fit(train_articles)
		#
		val_accuracy = E.evaluate(model=model, articles=val_articles, verbose=0)
		test_accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)
		# for curiosity, get the weights of the PPD dimensions
		coefs = model.model.coef_[0]
		coef_str = "\t".join([str(round(v, 3)) for v in coefs])
		res_str = "{}\t{}\t{}\t{:.1f}\t{:.1f}\t{:.1f}\t{}".format(sent_encoder.name, feature, quantiles, 100*proxy_val_acc, 100*val_accuracy, 100*test_accuracy, coef_str)
		print(res_str)
		results_file.write(res_str+"\n")
		results_file.flush()
		# if val acc > best, save model
		#if val_accuracy > max_val_acc:
		#	print("SAVING MODEL!")
		#	max_val_acc = val_accuracy
		#	sent_encoder.models[0].save("saved_models/PDStar_{}.h5".format(feature))



K.clear_session()

results_file.close()



'''
timestamp = time.strftime("%F-%H:%M:%S")
results_file = open("results/pdstar_stage_2_{}.txt".format(timestamp), "w")

chosen_features = []
max_val_score = 0

while True:#len(chosen_features) <= 5:

	best_feature = None
	best_val = 0

	for feature in h_encoder._all_features:
		if feature in chosen_features: continue

		#h_encoder.set_features(chosen_features+[feature])
		print("Features:", chosen_features+[feature])
		
		model_fnames = ["saved_models/PDStar_{}.h5".format(n) for n in chosen_features+[feature]]

		sent_encoder = PDStarMultiEncoder(sent_encoder=pre_sent_encoder, model_fnames=model_fnames)#layer_sizes=(256, 128), layer_dropouts=(0.5, 0.25))
		# due to computational constrains, we only actually fit on 75% of the training articles
		#sent_encoder.fit(train_articles=train_articles, val_articles=val_articles, verbose=0, max_epochs=1 if quick_mode else 100)

		#_ = input("?")
		#
		model = SimplePQModel(sent_encoder=sent_encoder, clf_type=LogisticRegression, clf_args={'class_weight':'balanced', 'max_iter':1000, 'solver':'lbfgs', 'C':0.1})
		model.fit(train_articles)
		#
		val_accuracy = E.evaluate(model=model, articles=val_articles, verbose=0)
		test_accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)
		# for curiosity, get the weights of the PPD dimensions
		if len(chosen_features+[feature]) == 1:
			coefs = model.model.coef_[0]
			coef_str = "\t".join([str(round(v, 3)) for v in coefs])
			res_str = "{}\t{}\t{:.1f}\t{:.1f}\t{}".format(sent_encoder.name, ', '.join(chosen_features+[feature]), 100*val_accuracy, 100*test_accuracy, coef_str)
		else:	
			res_str = "{}\t{}\t{:.1f}\t{:.1f}".format(sent_encoder.name, ', '.join(chosen_features+[feature]), 100*val_accuracy, 100*test_accuracy)
		print(res_str)
		results_file.write(res_str+"\n")
		results_file.flush()

		if val_accuracy > best_val:
			best_val = val_accuracy
			best_feature = feature

		K.clear_session()
	
	print("\n")
	results_file.write("\n")

	if best_val > max_val_score:
		max_val_score = best_val
		chosen_features.append(best_feature)
	else:
		break

	if len(chosen_features) >= len(h_encoder._all_features):
		break


K.clear_session()

results_file.close()
'''





