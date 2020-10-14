import numpy as np
import time
import random
import itertools
import argparse

from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, GRU, SimpleRNN, Bidirectional, Concatenate, RepeatVector, Subtract, Multiply, Dropout, Dot, Activation, Add, GaussianNoise
from keras.activations import sigmoid
import keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks

#from sklearn.linear_model import LogisticRegression

from models.sentence_encoders import SentBERTEncoder
#from models.SimplePQModel import SimplePQModel
#from models.PDStarMultiEncoder import PDStarMultiEncoder
#from models.PDStarEncoder import PDStarEncoder
from models.sentence_encoders import HandcraftedEncoder

import settings

from utils.pq_preprocessing import preprocess_pull_quotes
from utils.data_utils import get_article_partition
from utils.Evaluator import Evaluator


class TruePVDEncoder:

	def __init__(self, h_encoder, quantiles):
		self.name = "true_pvd_encoder"

		self.h_encoder = h_encoder
		self.quantiles = quantiles

	def fit(self, train_articles):

		feature_vals = []
		for a in train_articles:
			feature_vals.extend(self.feature_calculator(a))

		percentile_thresholds = np.percentile(feature_vals, [100*(i)/self.quantiles for i in range(1, self.quantiles)])

		def feature_to_onehot(value):
			onehot_vec = np.zeros(self.quantiles)
			bucket = np.sum(percentile_thresholds < value)
			onehot_vec[bucket] = 1
			return onehot_vec

		self.feature_to_onehot = feature_to_onehot

	def batch_encode(self, document):
		feature_vals = self.feature_calculator(document)

		encs = []

		for v in feature_vals:
			encs.append(self.feature_to_onehot(v))

		return np.array(encs)

	def feature_calculator(self, article):
		values = []
		sentences = article if type(article) == list else article['sentences']
		for sentence in sentences:
			values.append(self.h_encoder.encode(sentence, sentences)[0])
		return values



class GatedModel:

	def __init__(self, sent_encoder, pvd_encoder):

		self.name = "gated"

		self.sent_encoder = sent_encoder
		self.pvd_encoder = pvd_encoder

	def fit(self, train_articles, val_articles, max_epochs=100, verbose=0):

		K.clear_session()

		X_train = []
		gates_train = []
		y_train = []

		X_val = []
		gates_val = []
		y_val = []

		split_labels = ['train' for _ in train_articles]+['val' for _ in val_articles]

		for article, label in zip(train_articles+val_articles, split_labels):

			

			sentences = article['sentences']
			labels = article['inclusions']
			labels = [1 if v >= 1 else 0 for v in labels]

			sent_encs = [self.sent_encoder.encode(s) for s in sentences] #.astype(np.float32)

			if label == "train":
				X_train.extend(sent_encs)
			
				y_train.extend(labels)

				gates_train.extend(self.pvd_encoder.batch_encode(sentences))

			else:

				X_val.extend(sent_encs)
			
				y_val.extend(labels)

				gates_val.extend(self.pvd_encoder.batch_encode(sentences))

				
			
			#gates_train.extend()


		X_train = np.array(X_train).astype(np.float32)
		y_train = np.array(y_train).astype(np.int32)
		gates_train = np.array(gates_train).astype(np.float32)


		#gates_train = (gates_train.T / np.sum(gates_train, axis=1)).T

		X_val = np.array(X_val).astype(np.float32)
		y_val = np.array(y_val).astype(np.int32)
		gates_val = np.array(gates_val).astype(np.float32)

		#gates_train_initial = np.ones(gates_train.shape)/self.pvd_encoder.quantiles
		#gates_val_initial = np.ones(gates_val.shape)/self.pvd_encoder.quantiles
		gates_train_initial = np.random.uniform(0, 1, gates_train.shape)
		gates_val_initial = np.random.uniform(0, 1, gates_val.shape)
		gates_train_initial = (gates_train_initial.T / np.sum(gates_train_initial, axis=1)).T
		gates_val_initial = (gates_val_initial.T / np.sum(gates_val_initial, axis=1)).T

		self.model = self.get_model()

		callback = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

		#self.model.fit([X_train, gates_train_initial], y_train, epochs=max_epochs, batch_size=128, verbose=verbose, validation_data=([X_val, gates_val_initial], y_val), callbacks=[callback])

		self.model.fit([X_train, gates_train], y_train, epochs=max_epochs, batch_size=128, verbose=verbose, validation_data=([X_val, gates_val], y_val), callbacks=[callback])



	def get_model(self):

		input_shape = self.sent_encoder.shape

		nb_gates = self.pvd_encoder.quantiles

		main_input = Input(shape=input_shape, name='main_input')

		gating_input = Input(shape=(nb_gates,), name='gating_input')

		# = GaussianNoise(0.1)(gating_input)

		#merged = Concatenate()([main_input, gating_input])

		#x = Dense(128, activation='selu')(main_input)
		#x = Dropout(0.25)(x)

		many_predictions = Dense(nb_gates, activation='sigmoid')(main_input)


		main_output = Dot(axes=1)([many_predictions, gating_input])
		#main_output = Dense(1, activation='sigmoid')(merged)
		#main_output = Activation('sigmoid')(main_output)

		model = Model(inputs=[main_input, gating_input], outputs=main_output)

		model.compile(loss='binary_crossentropy', optimizer='adam')


		#doc_encoder = Model(inputs=doc_input, outputs=x_doc_mult)
		#doc_encoder.compile(loss='mse', optimizer='adam')

		return model

	def predict_article(self, sentences, document=None):

		sent_encs = np.array([self.sent_encoder.encode(s) for s in sentences]).astype(np.float32)

		gate_encs = self.pvd_encoder.batch_encode(sentences).astype(np.float32)


		y_pred = self.model.predict([sent_encs, gate_encs])
		
		y_pred = y_pred[:,0]

		return y_pred



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

h_encoder = HandcraftedEncoder(precomputed_embeddings=settings.PRECOMPUTED_HANDCRAFTED_EMBEDDINGS_FNAME)


#h_encoder.precompute(train_articles+val_articles+test_articles)

#import pickle
#with open("/home/tanner/ml_data/precomputed_handcrafted_embeddings.pkl", "wb") as f:
#	pickle.dump(h_encoder._precomputed, f)



timestamp = time.strftime("%F-%H:%M:%S")
results_file = open("results/gating_{}.txt".format(timestamp), "w")

quantile_options = [2]#, 5, 10, 20]


for feature in ["Sent_position", "Quote_count", "A_concreteness"]: #h_encoder._all_features[1:]:
	h_encoder.set_features(feature)
	print("Feature:", feature)
	#
	for quantiles in quantile_options:
		pvd_encoder = TruePVDEncoder(h_encoder=h_encoder, quantiles=quantiles)
		pvd_encoder.fit(train_articles)
		#
		model = GatedModel(sent_encoder=sent_encoder, pvd_encoder=pvd_encoder)
		model.fit(train_articles, val_articles, verbose=1, max_epochs=1 if quick_mode else 100)
		#
		val_accuracy = E.evaluate(model=model, articles=val_articles, verbose=0)
		test_accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)
		res_str = "{}\t{}\t{}\t{:.1f}\t{:.1f}".format(model.name, feature, quantiles, 100*val_accuracy, 100*test_accuracy)
		print(res_str)
		results_file.write(res_str+"\n")
		results_file.flush()



K.clear_session()

results_file.close()






