import numpy as np
import time
import random
import itertools
import argparse

import keras.backend as K

from sklearn.linear_model import LogisticRegression

from models.sentence_encoders import SentBERTEncoder
from models.SimplePQModel import SimplePQModel
from models.PPDEncoder import PPDEncoder

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



pre_sent_encoder = SentBERTEncoder(precomputed_embeddings=settings.PRECOMPUTED_SBERT_EMBEDDINGS_FNAME)


timestamp = time.strftime("%F-%H:%M:%S")
results_file = open("results/ppd_{}.txt".format(timestamp), "w")

'''
nb_trials = 1 if quick_mode else 5

quantile_options = [5, 10, 15, 20, 25, 30]
layer_size_options = [(128, 64), (256, 128), (512, 256)]
layer_dropout_options = [(0.5, 0.5), (0.5, 0.25), (0.25, 0.1), (0, 0)]

for trial in range(nb_trials):
	random.shuffle(train_articles)

	for quantiles, layer_sizes, layer_dropouts in itertools.product(quantile_options, layer_size_options, layer_dropout_options):
		sent_encoder = PPDEncoder(sent_encoder=pre_sent_encoder, quantiles=quantiles, layer_sizes=layer_sizes, layer_dropouts=layer_dropouts, activation='selu')
		
		# due to computational constrains, we only actually fit on 75% of the training articles
		sent_encoder.fit(train_articles[:int(len(train_articles)*0.75)], val_articles=val_articles, verbose=1)
		
		model = SimplePQModel(sent_encoder=sent_encoder, clf_type=LogisticRegression, clf_args={'class_weight':'balanced', 'max_iter':1000}, 'solver':'lbfgs')
		model.fit(train_articles)

		val_accuracy = E.evaluate(model=model, articles=val_articles, verbose=0)
		test_accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)

		# for curiosity, get the weights of the PPD dimensions
		coefs = model.model.coef_[0]
		coef_str = "\t".join([str(round(v, 3)) for v in coefs])

		res_str = "{}\t{}\t{}\t{}\t{}\t{:.1f}\t{:.1f}\t{}".format(trial, sent_encoder.name, quantiles, layer_sizes, layer_dropouts, 100*val_accuracy, 100*test_accuracy, coef_str)
		print(res_str)
		results_file.write(res_str+"\n")
		results_file.flush()
'''


for quantiles in [2, 5, 10, 20]:
	sent_encoder = PPDEncoder(sent_encoder=pre_sent_encoder, quantiles=quantiles, layer_sizes=(), layer_dropouts=(), activation='selu')
	
	# due to computational constrains, we only actually fit on 75% of the training articles
	sent_encoder.fit(train_articles, val_articles=val_articles, verbose=1)
	
	model = SimplePQModel(sent_encoder=sent_encoder, clf_type=LogisticRegression, clf_args={'class_weight':'balanced', 'max_iter':1000, 'solver':'lbfgs'})
	model.fit(train_articles)

	val_accuracy = E.evaluate(model=model, articles=val_articles, verbose=0)
	test_accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)

	# for curiosity, get the weights of the PPD dimensions
	coefs = model.model.coef_[0]
	coef_str = "\t".join([str(round(v, 3)) for v in coefs])

	res_str = "{}\t{}\t{}\t{:.1f}\t{:.1f}\t{}".format(trial, sent_encoder.name, quantiles, 100*val_accuracy, 100*test_accuracy, coef_str)
	print(res_str)
	results_file.write(res_str+"\n")
	results_file.flush()

K.clear_session()

results_file.close()




