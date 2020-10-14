
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Concatenate, RepeatVector, Subtract, Multiply, Dropout, Dot, Activation, Add
from keras.activations import sigmoid
import keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks


from sklearn.linear_model import LogisticRegression

import numpy as np
import math

class PDStarEncoder:


	def __init__(self, sent_encoder, feature_calculator, quantiles=10, layer_sizes=(256, 128), layer_dropouts=(0.5, 0.25), activation='selu'):
		self.name = "pd_star"

		# the high-dim sentence encoder (e.g. Sentence-BERT, fasttext)
		self.sent_encoder = sent_encoder

		# the feature we want to predict a distribution over (e.g. position)
		self.feature_calculator = feature_calculator

		self.quantiles = quantiles

		self.layer_sizes = layer_sizes
		self.layer_dropouts = layer_dropouts
		self.activation = activation


	def fit(self, train_articles, val_articles, verbose=1, max_epochs=100):
		K.clear_session()

		X_train = []
		y_train = []

		X_val = []
		y_val = []


		raw_feature_values_train = []
		raw_feature_values_val = []

		split_labels = ['train' for _ in train_articles]+['val' for _ in val_articles]

		for article, label in zip(train_articles+val_articles, split_labels):

			sentences = article['sentences']

			sent_encs = np.array([self.sent_encoder.encode(s) for s in sentences])

			feature_vals = self.feature_calculator(article)

			if label == 'train':
				X_train.extend(sent_encs)
				raw_feature_values_train.extend(feature_vals)
			else:
				X_val.extend(sent_encs)
				raw_feature_values_val.extend(feature_vals)


		# convert raw feature values to one-hot vectors
		nb_unique_values = len(set(raw_feature_values_train))
		self.quantiles = min(self.quantiles, nb_unique_values)
		print("nb_unique_values, quantiles:", nb_unique_values, self.quantiles)

		percentile_thresholds = np.percentile(raw_feature_values_train, [100*(i)/self.quantiles for i in range(1, self.quantiles)])
		print("Percentile thresholds:", percentile_thresholds)

		def feature_to_onehot(value):
			onehot_vec = np.zeros(self.quantiles)
			bucket = np.sum(percentile_thresholds < value)
			onehot_vec[bucket] = 1
			return onehot_vec


		for v in raw_feature_values_train:
			y_train.append(feature_to_onehot(v))

		for v in raw_feature_values_val:
			y_val.append(feature_to_onehot(v))


		X_train = np.array(X_train).astype(np.float32)
		y_train = np.array(y_train).astype(np.int32)

		
		X_val = np.array(X_val).astype(np.float32)
		y_val = np.array(y_val).astype(np.int32)

		callback = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

		self.model = get_model(X_train[0].shape, self.quantiles, self.layer_sizes, self.layer_dropouts, self.activation)
		
		fit_history = self.model.fit(X_train, y_train, epochs=max_epochs, batch_size=128, verbose=verbose, validation_data=(X_val, y_val), callbacks=[callback])
			
		#self.model = LogisticRegression(max_iter=1000, verbose=verbose, n_jobs=4, solver='lbfgs')
		#self.model.fit(X_train, y_train)
		return fit_history.history



	def encode(self, text, document=None):

		#sent_encs = np.array([self.sent_encoder.encode(s) for s in sentences]).astype(np.float32)
		sent_enc = self.sent_encoder.encode(text)
		ppd = self.model.predict(np.array([sent_enc]))[0]

		return ppd

	def batch_encode(self, texts, documents=None):
		sent_encs = np.array([self.sent_encoder.encode(s) for s in texts]).astype(np.float32)
		#sent_enc = self.sent_encoder.encode(text)
		ppds = self.model.predict(np.array(sent_encs))

		return ppds





def get_model(input_shape, output_dim, layer_sizes=(256, 256), layer_dropouts=(0.5, 0.25), activation='selu'):
	
	main_input = Input(shape=input_shape, name='main_input')

	x = main_input

	for i_l, (width, dropout) in enumerate(zip(layer_sizes, layer_dropouts)):
		x = Dense(width, activation=activation)(x)
		x = Dropout(dropout)(x)


	main_output = Dense(output_dim, activation='softmax')(x)

	model = Model(inputs=main_input, outputs=main_output)

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model


class QuantileEncoder:

	def __init__(self, quantiles):

		self.name = "true_quantile"

		self.quantiles = quantiles

	def batch_encode(self, sentences):

		n = len(sentences)

		encs = []

		for i in range(n):
			true_frac = i/(n-1)

			pos_onehot = np.zeros(self.quantiles)
			bucket = int(min(math.floor(true_frac * self.quantiles), self.quantiles-1))

			pos_onehot[bucket] = 1

			encs.append(pos_onehot)

		return np.array(encs)
