
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Concatenate, RepeatVector, Subtract, Multiply, Dropout, Dot, Activation, Add
from keras.activations import sigmoid
import keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks


from sklearn.linear_model import LogisticRegression

import numpy as np
import math

class PDStarMultiEncoder:


	def __init__(self, sent_encoder, feature_calculator=None, quantiles=10, layer_sizes=(256, 128), layer_dropouts=(0.5, 0.25), activation='selu', model_fnames=None):
		self.name = "pd_star"

		# the high-dim sentence encoder (e.g. Sentence-BERT, fasttext)
		self.sent_encoder = sent_encoder

		# the feature we want to predict a distribution over (e.g. position)
		self.feature_calculator = feature_calculator

		self.model_fnames = model_fnames

		self.layer_sizes = layer_sizes
		self.layer_dropouts = layer_dropouts
		self.activation = activation

		assert not (feature_calculator == None and model_fnames == None)

		if self.model_fnames != None:
			self.models = [load_model(fname) for fname in self.model_fnames]
			self.quantiles = [model.output_shape[1] for model in self.models]
		else:
			self.quantiles = quantiles

		


	def fit(self, train_articles, val_articles, verbose=1, max_epochs=100):
		K.clear_session()

		X_train = []
		X_val = []
		


		split_labels = ['train' for _ in train_articles]+['val' for _ in val_articles]

		for article, label in zip(train_articles+val_articles, split_labels):
			sentences = article['sentences']
			sent_encs = np.array([self.sent_encoder.encode(s) for s in sentences])

			if label == 'train':
				X_train.extend(sent_encs)
			else:
				X_val.extend(sent_encs)

		y_train = []
		y_val = []

		#for i, (feature_calculator, Q) in enumerat(zip(self.feature_calculators, self.quantiles)):

		raw_feature_values_train = []
		raw_feature_values_val = []

		for article, label in zip(train_articles+val_articles, split_labels):

			feature_vals = self.feature_calculator(article)

			if label == 'train':
				raw_feature_values_train.extend(feature_vals)
			else:
				raw_feature_values_val.extend(feature_vals)


		# convert raw feature values to one-hot vectors
		#nb_unique_values = len(set(raw_feature_values_train))
		#Q = min(self.quantiles, nb_unique_values)
		#print("nb_unique_values, quantiles:", nb_unique_values, Q)
		#self.quantiles[i] = Q

		raw_feature_values_train = np.array(raw_feature_values_train)
		raw_feature_values_val = np.array(raw_feature_values_val)

		nb_features = len(raw_feature_values_train[0])
		print("NB FEATURES = ", nb_features)

		for col_index in range(nb_features):
			values_train = raw_feature_values_train[:,col_index]
			values_val = raw_feature_values_val[:,col_index]

			percentile_thresholds = np.percentile(values_train, [100*(i)/self.quantiles for i in range(1, self.quantiles)])
			#print("Percentile thresholds:", percentile_thresholds)

			def feature_to_onehot(value):
				onehot_vec = np.zeros(self.quantiles)
				bucket = np.sum(percentile_thresholds < value)
				onehot_vec[bucket] = 1
				return onehot_vec

			y_feature_train = []
			y_feature_val = []

			for v in values_train:
				y_feature_train.append(feature_to_onehot(v))

			for v in values_val:
				y_feature_val.append(feature_to_onehot(v))

			y_train.append(np.array(y_feature_train).astype(np.int32))
			y_val.append(np.array(y_feature_val).astype(np.int32))


		X_train = np.array(X_train).astype(np.float32)
		X_val = np.array(X_val).astype(np.float32)

		#y_train = np.array(y_train).astype(np.int32)
		#y_val = np.array(y_val).astype(np.int32)

		callback = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

		self.models = [get_model(X_train[0].shape, self.quantiles, self.layer_sizes, self.layer_dropouts, self.activation) for _ in range(nb_features)]

		for i_model, model in enumerate(self.models):
			if verbose:
				print("Training model {}/{}".format(i_model+1, len(self.models)))
				print()
		
			model.fit(X_train, y_train[i_model], epochs=max_epochs, batch_size=128, verbose=verbose, validation_data=(X_val, y_val[i_model]), callbacks=[callback])
		
		#self.model = LogisticRegression(max_iter=1000, verbose=verbose, n_jobs=4, solver='lbfgs')
		#self.model.fit(X_train, y_train)




	def batch_encode(self, texts, documents=None):
		sent_encs = np.array([self.sent_encoder.encode(s) for s in texts]).astype(np.float32)
		#sent_enc = self.sent_encoder.encode(text)
		ppds = [model.predict(np.array(sent_encs)) for model in self.models]
		#if type(ppds) == list:
		ppds = np.concatenate(ppds, axis=1)

		return ppds





def get_model(input_shape, output_dims, layer_sizes=(256, 256), layer_dropouts=(0.5, 0.25), activation='selu'):
	
	if type(output_dims) == int:
		output_dims = [output_dims]

	nb_channels = len(output_dims)

	main_input = Input(shape=input_shape, name='main_input')

	outputs = []

	for i in range(nb_channels):
		x = main_input

		for i_l, (width, dropout) in enumerate(zip(layer_sizes, layer_dropouts)):
			x = Dense(width, activation=activation)(x)
			x = Dropout(dropout)(x)

		output = Dense(output_dims[i], activation='softmax')(x)

		outputs.append(output)

	model = Model(inputs=main_input, outputs=outputs)

	model.compile(loss=['categorical_crossentropy' for _ in outputs], optimizer='adam')

	return model

