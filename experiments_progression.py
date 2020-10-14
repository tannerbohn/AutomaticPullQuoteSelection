import time
import argparse
import itertools

import keras.backend as K

from models.sentence_encoders import SentBERTEncoder
from models.FlexiblePQModel import FlexiblePQModel

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



sent_encoder = SentBERTEncoder(precomputed_embeddings=settings.PRECOMPUTED_SBERT_EMBEDDINGS_FNAME)


timestamp = time.strftime("%F-%H:%M:%S")
results_file = open("results/progression_{}.txt".format(timestamp), "w")


model = FlexiblePQModel(sent_encoder=sent_encoder)

# prepare data
model.prepare_data(train_articles, val_articles)

# modes: v1, v1_deep, v2, v2_deep, v3, v3_deep

trials = 1 if quick_mode else 5


model.mode = "A_basic"
for trial in range(trials):
	model.train_model()

	val_accuracy = E.evaluate(model=model, articles=val_articles, verbose=0)
	test_accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)
	nb_params = model.model.count_params()
	res_str = "{}\t{}\t{}\t{}\t{}\t{}\t{:.1f}\t{:.1f}".format(trial, model.name, model.mode, "-", "-", nb_params,  100*val_accuracy, 100*test_accuracy)
	print(res_str)
	results_file.write(res_str+"\n")
	results_file.flush()


model.mode = "A_deep"
for width in [16, 32, 64, 128, 256, 512]:
	for trial in range(trials):
		model.train_model(width=width)

		val_accuracy = E.evaluate(model=model, articles=val_articles, verbose=0)
		test_accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)
		nb_params = model.model.count_params()
		res_str = "{}\t{}\t{}\t{}\t{}\t{}\t{:.1f}\t{:.1f}".format(trial, model.name, model.mode, width, "-",  nb_params, 100*val_accuracy, 100*test_accuracy)
		print(res_str)
		results_file.write(res_str+"\n")
		results_file.flush()


########################################
model.mode = "B_basic"

for trial in range(trials):
	model.train_model()

	val_accuracy = E.evaluate(model=model, articles=val_articles, verbose=0)
	test_accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)
	nb_params = model.model.count_params()
	res_str = "{}\t{}\t{}\t{}\t{}\t{}\t{:.1f}\t{:.1f}".format(trial, model.name, model.mode, "-", "-",  nb_params, 100*val_accuracy, 100*test_accuracy)
	print(res_str)
	results_file.write(res_str+"\n")
	results_file.flush()


model.mode = "B_deep"

for width in [16, 32, 64, 128, 256, 512]:
	for trial in range(trials):
		model.train_model(width=width)

		val_accuracy = E.evaluate(model=model, articles=val_articles, verbose=0)
		test_accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)
		nb_params = model.model.count_params()
		res_str = "{}\t{}\t{}\t{}\t{}\t{}\t{:.1f}\t{:.1f}".format(trial, model.name, model.mode, width, "-", nb_params,  100*val_accuracy, 100*test_accuracy)
		print(res_str)
		results_file.write(res_str+"\n")
		results_file.flush()

########################################

model.mode = "C_basic"
for nb_experts in [2, 4, 8, 16]:
	for trial in range(trials):
		model.train_model(nb_experts=nb_experts)

		val_accuracy = E.evaluate(model=model, articles=val_articles, verbose=0)
		test_accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)
		nb_params = model.model.count_params()
		res_str = "{}\t{}\t{}\t{}\t{}\t{}\t{:.1f}\t{:.1f}".format(trial, model.name, model.mode, "-", nb_experts,  nb_params, 100*val_accuracy, 100*test_accuracy)
		print(res_str)
		results_file.write(res_str+"\n")
		results_file.flush()



model.mode = "C_deep"
for width, nb_experts in itertools.product([16, 32, 64, 128, 256, 512], [2, 4, 8, 16]):
	for trial in range(trials):
		model.train_model(nb_experts=nb_experts, width=width)
		val_accuracy = E.evaluate(model=model, articles=val_articles, verbose=0)
		test_accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)
		nb_params = model.model.count_params()
		res_str = "{}\t{}\t{}\t{}\t{}\t{}\t{:.1f}\t{:.1f}".format(trial, model.name, model.mode, width, nb_experts, nb_params, 100*val_accuracy, 100*test_accuracy)
		print(res_str)
		results_file.write(res_str+"\n")
		results_file.flush()


K.clear_session()
results_file.close()
