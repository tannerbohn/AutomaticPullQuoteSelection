
from collections import Counter
import numpy as np

import settings

from utils.pq_preprocessing import preprocess_pull_quotes
from utils.data_utils import get_article_partition


articles_data = preprocess_pull_quotes(directories=settings.PQ_SAMPLES_DIRS)
#articles_data = articles_data[:100]
print("# articles = ", len(articles_data))

train_articles, val_articles, test_articles = get_article_partition(articles_data, train_frac=0.7, val_frac=0.1, test_frac=0.2, seed=1337, verbose=1)


news_sources = [a['url'].replace("https://", "").replace("http://", "").replace("www.", "").split(".com")[0] for a in articles_data]
news_sources = [s if s != "business.financialpost" else "nationalpost" for s in news_sources]


source_grouped_articles = dict()

for article, source in zip(articles_data, news_sources):
	try:
		source_grouped_articles[source].append(article)
	except:
		source_grouped_articles[source] = [article]

source_grouped_articles['all'] = articles_data
source_grouped_articles['train'] = train_articles
source_grouped_articles['val'] = val_articles
source_grouped_articles['test'] = test_articles

for group in source_grouped_articles.keys():
	articles = source_grouped_articles[group]

	total_nb_articles = len(articles)

	pq_per_article = [a['nb_pq'] for a in articles]

	total_pq = sum(pq_per_article)
	avg_pq_per_article = np.average(pq_per_article)

	doc_lengths = [len(a['sentences']) for a in articles]
	avg_doc_length = np.average(doc_lengths)

	nb_pos_samples = sum([len(a['inclusions']) - a['inclusions'].count(0) for a in articles])
	nb_neg_samples = sum([a['inclusions'].count(0) for a in articles])

	sents_per_pq = []
	for a in articles:
		C = Counter(a['inclusions'])
		del C[0]
		sents_per_pq.extend(list(C.values()))

	avg_sent_per_pq = np.average(sents_per_pq)

	print(f"{group}\t{total_nb_articles}\t{total_pq}\t{avg_pq_per_article:.2f}\t{avg_sent_per_pq:.2f}\t{avg_doc_length:.2f}\t{nb_pos_samples}\t{nb_neg_samples}")
