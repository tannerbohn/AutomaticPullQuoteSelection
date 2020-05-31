import numpy as np
from sklearn.metrics import roc_auc_score

class Evaluator:

	def __init__(self):
		return


	def evaluate(self, model, articles, verbose=0):

		scores = []

		for i_article, article in enumerate(articles):
			if verbose and i_article %100 == 0:
				if verbose: print("Evaluation: {:.2f}%".format(100*i_article/len(articles)))

			labels = article['inclusions']

			# binarize the labels (the inclusion vectors use different integers for the different PQs in an article)
			y_true = np.array([1 if v >= 1 else 0 for v in labels])

			y_pred = model.predict_article(article['sentences'], document=article['sentences'])

			# 50% = random guess
			score = roc_auc_score(y_true, y_pred)
			scores.append(score)

		return np.average(scores)