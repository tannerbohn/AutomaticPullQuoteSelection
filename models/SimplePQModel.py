



import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

class SimplePQModel:

	def __init__(self, sent_encoder, doc_normalize=False, enc_dim="orig", clf_type=LogisticRegression, clf_args={}, dtype=np.float32):

		self.name = "simple"
		self.sent_encoder = sent_encoder


		self.doc_normalize = doc_normalize
		self.enc_dim = enc_dim

		self._clf_type = clf_type
		self._clf_args = clf_args

		self.dtype = dtype


		return

	def reset(self):
		self.sent_encoder.reset()
		return


	def fit(self, articles, neg_radius=3):


		X_docs = []

		y = []

		for i_article, article in enumerate(articles):
			#if i_article % 500 == 0: print("\t{:.2f}".format(100*i_article/len(articles)))

			#doc_enc = self.sent_encoder.encode(article['sentences'])

			sentences = article['sentences']
			labels = article['inclusions']
			'''
			for i in range(len(article['sentences'])):
				start_index = max(0, i-neg_radius)
				end_index = i+neg_radius
				if any(article['inclusions'][start_index:end_index+1]):
					sentences.append(article['sentences'][i])
					labels.append(article['inclusions'][i])
			'''

			
			if self.sent_encoder.name == "ppd":
				sent_encs = self.sent_encoder.batch_encode(sentences)
			elif self.sent_encoder.name == "true_quantile":
				sent_encs = self.sent_encoder.batch_encode(sentences)
			else:
				sent_encs = np.array([self.sent_encoder.encode(s, document=sentences) for s in sentences])

			y_doc = [1 if v >= 1 else 0 for v in labels]

			if self.doc_normalize:
				doc_mean = np.mean(sent_encs, axis=0)
				doc_std = np.std(sent_encs, axis=0)
				doc_std[doc_std==0] = 1
				sent_encs = (sent_encs - doc_mean)/doc_std

			sent_encs = sent_encs.astype(self.dtype)

			X_docs.extend(sent_encs)
			y.extend(y_doc)


		# scale data
		#self.scaler = StandardScaler().fit(X_docs)
		#X_docs = self.scaler.transform(X_docs)

		if self.enc_dim != "orig":
			self.transformer = SparseRandomProjection(n_components=self.enc_dim, dense_output=True)
			#self.transformer = GaussianRandomProjection(n_components=self.enc_dim)
			X_docs = self.transformer.fit_transform(X_docs)

		self.model = self._clf_type(**self._clf_args)

		self.model.fit(X_docs, y)


		return

	def predict_article(self, sentences, document=None):

		#sent_encs = np.array([self.sent_encoder.encode(s, document=document) for s in sentences])

		if self.sent_encoder.name == "ppd":
			sent_encs = self.sent_encoder.batch_encode(sentences)
		elif self.sent_encoder.name == "true_quantile":
			assert len(document) == len(sentences)
			sent_encs = self.sent_encoder.batch_encode(document)
		else:
			sent_encs = np.array([self.sent_encoder.encode(s, document=document) for s in sentences])

		if self.doc_normalize:

			if document != None and len(sentences) == len(document):
				all_sent_encs = sent_encs
			else:
				all_sent_encs = np.array([self.sent_encoder.encode(s) for s in document])
			
			doc_mean = np.mean(all_sent_encs, axis=0)
			doc_std = np.std(all_sent_encs, axis=0)
			doc_std[doc_std==0] = 1

			sent_encs = (sent_encs - doc_mean)/doc_std

		sent_encs = sent_encs.astype(self.dtype)

		if self.enc_dim != "orig":
			sent_encs = self.transformer.transform(sent_encs)

		#sent_encs = self.scaler.transform(sent_encs)

		y_pred = self.model.predict_proba(sent_encs)[:,1]


		return y_pred
