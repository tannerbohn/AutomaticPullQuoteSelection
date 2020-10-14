import time
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from models.sentence_encoders import SentBERTEncoder
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



sent_encoder = SentBERTEncoder(precomputed_embeddings=settings.PRECOMPUTED_SBERT_EMBEDDINGS_FNAME)


timestamp = time.strftime("%F-%H:%M:%S")
results_file = open("results/sbert_{}.txt".format(timestamp), "w")

'''
print("Training main model")
for trial in range(5):
	for enc_dim in [2, 5, 10, 20]:
		model = SimplePQModel(sent_encoder=sent_encoder, enc_dim=enc_dim, clf_type=LogisticRegression, clf_args={'class_weight':'balanced', 'max_iter':1000})
		model.fit(train_articles)
		#print("Done")
		#for refinement_size in [1, 2, 3, 4, 5, 6]:
		#	print("training refinement model...")
		#	model.post_model = None
		#	model.fit_refinement_model(train_articles, refinement_size, clf_type=AdaBoostClassifier, clf_args={'n_estimators':100, 'base_estimator':DecisionTreeClassifier(max_depth=3, class_weight="balanced")})
		#	print("evaluating...")
		accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)
		res_str = "{}\t{}\t{}\t{:.1f}".format(trial, sent_encoder.name, enc_dim ,100*accuracy)
		print(res_str)
		#results_file.write(res_str+"\n")
'''




model = SimplePQModel(sent_encoder=sent_encoder, clf_type=LogisticRegression, clf_args={'class_weight':'balanced', 'max_iter':1000})
model.fit(train_articles)
accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)
res_str = "{}\t{:.1f}".format(sent_encoder.name ,100*accuracy)
print(res_str)

'''
results = []
for i_a, a in enumerate(test_articles):
	auc = E.evaluate(model=model, articles=[a])
	results.append((a, auc))

results = sorted(results, key=lambda el: el[1])
for a, auc in results[:100]: 
	source_sentences = [s for i_s, s in enumerate(a['sentences']) if a['inclusions'][i_s] > 0]
	#print("{:.3f}\t{}\n\t{}".format(100*auc, a['edited_pqs'], a['url']))
	print("{:.3f}\t{}\n\t{}".format(100*auc, source_sentences, a['edited_pqs']))
	print()
'''
#results_file.close()
