import time
import argparse
import numpy as np

from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from models.sentence_encoders import SentBERTEncoder
from models.SimplePQModel import SimplePQModel

import settings

from utils.pq_preprocessing import preprocess_pull_quotes
from utils.data_utils import get_article_partition
from utils.Evaluator import Evaluator


from sklearn.feature_extraction.text import TfidfVectorizer


def get_dimension_meanings(sentences, sent_encs, dim_index, k=1000):
	sent_values = sent_encs[:,dim_index]
	sorted_sents, sorted_scores = zip(*sorted(zip(sentences, sent_values), key=lambda el: el[1]))
	#sorted_sents = sorted(zip(sent_values, sentences), key=lambda el: el[0])
	# get overlap top top k sentences
	top_sents = sorted_sents[-k:]
	bottom_sents = sorted_sents[:k]
	nb_sents = len(sentences)
	middle_sents = sorted_sents[-nb_sents//2-k:-nb_sents//2+k]
	vectorizer = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf = True, stop_words='english')
	vectors = vectorizer.fit_transform([' '.join(top_sents), ' '.join(middle_sents), ' '.join(bottom_sents)])
	feature_names = vectorizer.get_feature_names()
	dense = vectors.todense()
	denselist = dense.tolist()
	#
	top_term_ratings = list(zip(feature_names, denselist[0]))
	top_terms = list(zip(*sorted(top_term_ratings, key=lambda el: -el[1])))[0][:20]
	#
	bottom_term_ratings = list(zip(feature_names, denselist[2]))
	bottom_terms = list(zip(*sorted(bottom_term_ratings, key=lambda el: -el[1])))[0][:20]
	# need to replace the '"' character, otherwise it won't appear properly in the google spreadsgeet (for some unknown reason)
	return top_terms, bottom_terms, sorted_sents[-1].replace('"', "“"), sorted_sents[0].replace('"', "“")#, top_terms, bottom_terms






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
results_file = open("results/sbert_dims_{}.txt".format(timestamp), "w")


model = SimplePQModel(sent_encoder=sent_encoder, clf_type=LogisticRegression, clf_args={'class_weight':'balanced', 'max_iter':1000})


all_sentences = []
for a in train_articles:
	all_sentences.extend(a['sentences'])

sent_encoder.set_dimensions("all")
sent_encs = np.array([sent_encoder.encode(s) for s in all_sentences])

for dim in tqdm(list(range(768))):
	sent_encoder.set_dimensions(dim)
	model.fit(train_articles)
	accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)
	coef = model.model.coef_[0][0]
	top_terms, bottom_terms, top_sentence, bottom_sentence = get_dimension_meanings(all_sentences, sent_encs, dim, k=2000 if not quick_mode else 15)
	#accuracy, coef = 1, 1
	res_str = "{}\t{}\t{:.1f}\t{:.4f}\t{}\t{}\t{}\t{}".format(sent_encoder.name, dim, 100*accuracy, coef, top_terms, bottom_terms, top_sentence, bottom_sentence)
	#print(res_str)

	results_file.write(res_str+"\n")
	results_file.flush()

results_file.close()


