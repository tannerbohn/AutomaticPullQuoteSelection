import time
import argparse

import settings
from utils.pq_preprocessing import preprocess_pull_quotes
from utils.data_utils import get_article_partition
from utils.Evaluator import Evaluator

from models.sentence_encoders import SentBERTEncoder
from models.ClickbaitPQModel import ClickbaitPQModel
from models.HeadlinePopularityPQModel import HeadlinePopularityPQModel
from models.SummarizerPQModel import SummarizerPQModel




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
results_file = open("results/cross_task_{}.txt".format(timestamp), "w")


sent_encoder = SentBERTEncoder(precomputed_embeddings=settings.PRECOMPUTED_SBERT_EMBEDDINGS_FNAME)


print("Running clickbait experiment...")
model = ClickbaitPQModel(sent_encoder=sent_encoder, clickbait_fname=settings.CLICKBAIT_FNAME, headlines_fname=settings.HEADLINE_FNAME, quick_mode=quick_mode)
model.fit(train_articles)
accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)
res_str = "{}\t{:.1f}".format(model.name, 100*accuracy)
print(res_str)
results_file.write(res_str+"\n")
results_file.flush()


print("Running headline experiment...")
model = HeadlinePopularityPQModel(sent_encoder=sent_encoder, dataset_fname=settings.HEADLINE_POPULARITY_FNAME, quick_mode=quick_mode)
model.fit(train_articles)
accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)
res_str = "{}\t{:.1f}".format(model.name, 100*accuracy)
print(res_str)
results_file.write(res_str+"\n")
results_file.flush()




print("Running summarizer experiment...")
for name in ["LexRankSummarizer", "SumBasicSummarizer", "KLSummarizer", "TextRankSummarizer"]:
	model = SummarizerPQModel(name=name)
	accuracy = E.evaluate(model, test_articles, verbose=1)
	res_str = "{}\t{:.1f}".format(model.name, 100*accuracy)
	print(res_str)
	results_file.write(res_str+"\n")
	results_file.flush()



results_file.close()