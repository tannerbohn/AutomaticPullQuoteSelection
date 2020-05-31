
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Concatenate, RepeatVector, Subtract, Multiply, Dropout, Dot, Activation, Add
from keras.activations import sigmoid
import keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks

import numpy as np
import math

class PPDEncoder:


	def __init__(self, sent_encoder, quantiles=10, layer_sizes=(256, 256), layer_dropouts=(0.5, 0.25), activation='selu'):
		self.name = "ppd"
		self.sent_encoder = sent_encoder

		self.quantiles = quantiles
		self.layer_sizes = layer_sizes
		self.layer_dropouts = layer_dropouts
		self.activation = activation


	def fit(self, train_articles, val_articles, verbose=1):
		K.clear_session()

		X_train = []
		y_train = []

		X_val = []
		y_val = []

		split_labels = ['train' for _ in train_articles]+['val' for _ in val_articles]

		for article, label in zip(train_articles+val_articles, split_labels):

			sentences = article['sentences']

			sent_encs = np.array([self.sent_encoder.encode(s) for s in sentences])

			if label == 'train':
				X_train.extend(sent_encs)
			else:
				X_val.extend(sent_encs)

			n = len(sentences)

			for i in range(n):
				true_frac = i/(n-1)

				pos_onehot = np.zeros(self.quantiles)
				bucket = int(min(math.floor(true_frac * self.quantiles), self.quantiles-1))

				pos_onehot[bucket] = 1

				if label == 'train':
					y_train.append(pos_onehot)
				else:
					y_val.append(pos_onehot)

		X_train = np.array(X_train).astype(np.float32)
		y_train = np.array(y_train).astype(np.int32)

		X_val = np.array(X_val).astype(np.float32)
		y_val = np.array(y_val).astype(np.int32)


		callback = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

		self.model = get_model(X_train[0].shape, self.quantiles, self.layer_sizes, self.layer_dropouts, self.activation)
		
		self.model.fit(X_train, y_train, epochs=30, batch_size=128, verbose=verbose, validation_data=(X_val, y_val), callbacks=[callback])

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

	model.compile(loss='categorical_crossentropy', optimizer='adam')

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
