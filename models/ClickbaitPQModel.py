import numpy as np
from sklearn.linear_model import LogisticRegression

class ClickbaitPQModel:

	def __init__(self, sent_encoder, clickbait_fname, headlines_fname, quick_mode=False):

		self.name = "clickbait"

		self.sent_encoder = sent_encoder

		self.clickbait_samples = []
		self.headline_samples = []

		with open(clickbait_fname, "r") as f:
			lines = f.read().splitlines()
			lines = [l.strip() for l in lines if l.strip() != ""]
			self.clickbait_samples = lines

		with open(headlines_fname, "r") as f:
			lines = f.read().splitlines()
			lines = [l.strip() for l in lines if l.strip() != ""]
			self.headline_samples = lines

		if quick_mode:
			self.clickbait_samples = self.clickbait_samples[:100]
			self.headline_samples = self.headline_samples[:100]


		print("Encoding clickbait...")
		clickbait_X = self.sent_encoder.batch_encode(self.clickbait_samples, verbose=1)
		print("Encoding headlines...")
		headlines_X = self.sent_encoder.batch_encode(self.headline_samples, verbose=1)
		self.y = np.array([1 for _ in clickbait_X]+[0 for _ in headlines_X])
		self.X = np.array(clickbait_X+headlines_X)
		print("done")

		return


	def fit(self, articles):


		self.model = LogisticRegression()

		
		self.model.fit(self.X, self.y)

	def predict_article(self, sentences, document=None):
		
		# for each sentence get clickbait prob
		sent_encs = [self.sent_encoder.encode(s) for s in sentences]
		y_pred = self.model.predict_proba(sent_encs)[:,1]

		return y_pred