import time
import argparse

from sklearn.linear_model import LogisticRegression

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


model = SimplePQModel(sent_encoder=sent_encoder, clf_type=LogisticRegression, clf_args={'class_weight':'balanced', 'max_iter':1000})
model.fit(train_articles)
accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)
res_str = "{}\t{:.1f}".format(sent_encoder.name, 100*accuracy)
print(res_str)

results_file.write(res_str+"\n")
results_file.close()
