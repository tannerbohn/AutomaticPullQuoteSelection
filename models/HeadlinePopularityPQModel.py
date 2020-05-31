
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from sklearn.linear_model import LinearRegression

class HeadlinePopularityPQModel:

	def __init__(self, sent_encoder, dataset_fname, quick_mode=False):

		self.name = "headline_popularity"

		self.sent_encoder = sent_encoder

		main_data = pd.read_csv(dataset_fname)
		# Grab all the article titles
		article_titles = main_data['Title']
		#article_titles.shape
		#titles_list = [title for title in article_titles]

		facebook_scores = np.array(main_data['Facebook'])
		googleplus_scores = np.array(main_data['GooglePlus'])
		linkedin_scores = np.array(main_data['LinkedIn'])

		# combine into final data and scores
		titles = []
		scores = []
		for i_t, (title, score_f, score_g, score_l) in enumerate(zip(article_titles, facebook_scores, googleplus_scores, linkedin_scores)):
			if i_t % 1000 == 0:
				print("{:.3f}%".format(100*i_t/len(article_titles)))

			if quick_mode and i_t > 100: break

			valid_percentiles = []
			if score_f != -1:
				p = percentileofscore(facebook_scores, score_f)
				valid_percentiles.append(p)
			if score_g != -1:
				p = percentileofscore(googleplus_scores, score_g)
				valid_percentiles.append(p)
			if score_l != -1:
				p = percentileofscore(linkedin_scores, score_l)
				valid_percentiles.append(p)
			if len(valid_percentiles) == 0:
				#print("no valid!")
				continue
			else:
				titles.append(title)
				scores.append(np.average(valid_percentiles)/100)

		self.X = self.sent_encoder.batch_encode(titles, verbose=1)
		self.y = np.array(scores)

		return

	def fit(self, articles):

		self.model = LinearRegression()
		self.model.fit(self.X, self.y)

	def predict_article(self, sentences, document=None):
		
		# for each sentence get clickbait prob
		sent_encs = [self.sent_encoder.encode(s) for s in sentences]
		y_pred = self.model.predict(sent_encs)
		y_pred /= np.max(y_pred)

		return y_pred