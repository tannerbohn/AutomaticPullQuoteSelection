
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, GRU, SimpleRNN, Bidirectional, Concatenate, RepeatVector, Subtract, Multiply, Dropout, Dot, Activation, Add
from keras.activations import sigmoid
import keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks


import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB



from sklearn.preprocessing import StandardScaler

from sklearn.semi_supervised import LabelPropagation

from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection


class FlexiblePQModel:

	def __init__(self, sent_encoder, mode="A-basic"):

		self.name = "flexible"
		self.sent_encoder = sent_encoder

		self.mode=mode


		return

	def reset(self):
		self.sent_encoder.reset()
		return

	def set_mode(self, new_mode):

		self.mode = new_mode




	def prepare_data(self, train_articles, val_articles):

		# for each article, compute the document embedding and then store the learned paramaters
		# next, train a model to predict parameters given the embedding (return a list of embeddings
		#	maybe so that they can be clustered?)

		K.clear_session()

		#neg_radius = 3

		X_train = []
		context_train = []
		y_train = []

		X_val = []
		context_val = []
		y_val = []

		split_labels = ["train" for _ in train_articles]+["val" for _ in val_articles]

		for article, split_label in zip(train_articles+val_articles, split_labels):
			#if i_article % 500 == 0: print("\t{:.2f}".format(100*i_article/len(articles)))

			sentences = article['sentences']

			labels = article['inclusions']
			labels = [1 if v >= 1 else 0 for v in labels]


			sent_encs = [self.sent_encoder.encode(s) for s in sentences]
			context_enc = np.average(sent_encs, axis=0)


			

			if split_label == "train":
				X_train.extend(sent_encs)
				context_train.extend([context_enc for _ in sent_encs])
				y_train.extend(labels)
			else:
				X_val.extend(sent_encs)
				context_val.extend([context_enc for _ in sent_encs])
				y_val.extend(labels)


		self.X_train = np.array(X_train).astype(np.float32)
		self.context_train = np.array(context_train).astype(np.float32)
		self.y_train = np.array(y_train)

		self.X_val = np.array(X_val).astype(np.float32)
		self.context_val = np.array(context_val).astype(np.float32)
		self.y_val = np.array(y_val)


		

	def train_model(self, width=128, max_epochs=100, batch_size=128, verbose=1, regularization=(0,0), pos_weight=1, activation='selu', nb_experts=2, dropout=0.5):

		K.clear_session()

		callback = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

		if self.mode == "A_basic":
			# logistic regression
			self.model, self.doc_encoder = initialize_network_A_basic(input_shape=self.X_train[0].shape, regularization=regularization)
			self.model.fit(self.X_train, self.y_train, epochs=max_epochs, batch_size=batch_size, verbose=verbose, validation_data=(self.X_val, self.y_val), class_weight={0: 1, 1: pos_weight}, callbacks=[callback])

		elif self.mode == "B_basic":
			# lr with doc input as well
			self.model, self.doc_encoder = initialize_network_B_basic(input_shape=self.X_train[0].shape, regularization=regularization)
			self.model.fit([self.X_train, self.context_train], self.y_train, epochs=max_epochs, batch_size=batch_size, verbose=verbose, validation_data=([self.X_val, self.context_val], self.y_val), class_weight={0: 1, 1: pos_weight}, callbacks=[callback])

		elif self.mode == "C_basic":
			# basic meta-lr
			self.model, self.doc_encoder = initialize_network_C_basic(input_shape=self.X_train[0].shape, nb_experts=nb_experts, regularization=regularization)
			self.model.fit([self.X_train, self.context_train], self.y_train, epochs=max_epochs, batch_size=batch_size, verbose=verbose, validation_data=([self.X_val, self.context_val], self.y_val), class_weight={0: 1, 1: pos_weight}, callbacks=[callback])


		elif self.mode == "A_deep":
			# logistic regression
			self.model, self.doc_encoder = initialize_network_A_deep(input_shape=self.X_train[0].shape, width=width, regularization=regularization)
			self.model.fit(self.X_train, self.y_train, epochs=max_epochs, batch_size=batch_size, verbose=verbose, validation_data=(self.X_val, self.y_val), class_weight={0: 1, 1: pos_weight}, callbacks=[callback])

		elif self.mode == "B_deep":
			# lr with doc input as well
			self.model, self.doc_encoder = initialize_network_B_deep(input_shape=self.X_train[0].shape, width=width, regularization=regularization)
			self.model.fit([self.X_train, self.context_train], self.y_train, epochs=max_epochs, batch_size=batch_size, verbose=verbose, validation_data=([self.X_val, self.context_val], self.y_val), class_weight={0: 1, 1: pos_weight}, callbacks=[callback])

		elif self.mode == "C_deep":
			# basic meta-lr
			self.model, self.doc_encoder = initialize_network_C_deep(input_shape=self.X_train[0].shape, width=width, nb_experts=nb_experts, dropout=dropout)
			self.model.fit([self.X_train, self.context_train], self.y_train, epochs=max_epochs, batch_size=batch_size, verbose=verbose, validation_data=([self.X_val, self.context_val], self.y_val), class_weight={0: 1, 1: pos_weight}, callbacks=[callback])

		#pos_frac = np.average(self.y)
		#print("inv pos frac:", 1/pos_frac)
		#pos_weight = pos_weight/pos_frac
		#print("pos weight:", pos_weight)

		#self.model.fit([self.X_docs, self.doc_encs], self.y, epochs=epochs, batch_size=batch_size, verbose=verbose, class_weight={0: 1, 1: pos_weight})

		#return self.doc_encoder.predict(self.doc_encs)	


	def predict_article(self, sentences, document=None):

		sent_encs = np.array([self.sent_encoder.encode(s) for s in sentences]).astype(np.float32)

		context_enc = np.average(sent_encs, axis=0).astype(np.float32)
		#X_doc = np.array([self.sent_encoder.encode(s, document=document) for s in sentences]).astype(np.float32)


		if "B_" in self.mode or "C_" in self.mode:
			y_pred = self.model.predict([sent_encs, np.array([context_enc for _ in sent_encs])])
		else:
			y_pred = self.model.predict(sent_encs)

		y_pred = y_pred[:,0]

		return y_pred



###############################################################################

def initialize_network_A_basic(input_shape, regularization=(0,0)):
	# A_basic essentially implements logistic regression with only the candidate text as input

	main_input = Input(shape=input_shape, name='main_input')

	main_output = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=regularization[0], l2=regularization[1]))(main_input)

	model = Model(inputs=main_input, outputs=main_output)

	model.compile(loss='binary_crossentropy', optimizer='adam')

	return model, None

def initialize_network_A_deep(input_shape, width=256, regularization=(0,0), dropout=0.5):
	# A_basic essentially implements logistic regression with only the candidate text as input

	main_input = Input(shape=input_shape, name='main_input')

	x = Dense(width, activation='selu')(main_input)
	x = Dropout(dropout)(x)
	x = Dense(width//2, activation='selu')(x)
	#x = Dropout(dropout//2)(x)

	main_output = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=regularization[0], l2=regularization[1]))(x)

	model = Model(inputs=main_input, outputs=main_output)

	model.compile(loss='binary_crossentropy', optimizer='adam')

	return model, None

###############################################################################

def initialize_network_B_basic(input_shape, regularization=(0,0)):
	# this version simply concatenates the candidate and doc encodigs for logistic regression

	main_input = Input(shape=input_shape, name='main_input')

	doc_input = Input(shape=input_shape, name='doc_input')

	
	x = Concatenate()([main_input, doc_input])

	main_output = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=regularization[0], l2=regularization[1]))(x)

	model = Model(inputs=[main_input, doc_input], outputs=main_output)

	model.compile(loss='binary_crossentropy', optimizer='adam')


	return model, None

def initialize_network_B_deep(input_shape, width=256, regularization=(0,0), dropout=0.5):
	# this version simply concatenates the candidate and doc encodigs for logistic regression

	main_input = Input(shape=input_shape, name='main_input')

	doc_input = Input(shape=input_shape, name='doc_input')

	
	x = Concatenate()([main_input, doc_input])

	x = Dense(width, activation='selu')(x)
	x = Dropout(dropout)(x)
	x = Dense(width//2, activation='selu')(x)
	#x = Dropout(dropout//2)(x)

	main_output = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=regularization[0], l2=regularization[1]))(x)

	model = Model(inputs=[main_input, doc_input], outputs=main_output)

	model.compile(loss='binary_crossentropy', optimizer='adam')


	return model, None

###############################################################################


def initialize_network_C_basic(input_shape, nb_experts=2, regularization=(0,0)):
	# this is the basic version of the meta-lr

	main_input = Input(shape=input_shape, name='main_input')

	doc_input = Input(shape=input_shape, name='doc_input')

	x_doc_mult = Dense(nb_experts, activation='softmax', activity_regularizer=regularizers.l1_l2(l1=regularization[0], l2=regularization[1]))(doc_input) #, kernel_regularizer=regularizers.l1_l2(l1=regularization[0], l2=regularization[1]))(doc_input)

	sent_enc = Dense(nb_experts, activation='sigmoid')(main_input)

	main_output = Dot(axes=1)([sent_enc, x_doc_mult])
	#main_output = Activation('sigmoid')(main_output)

	model = Model(inputs=[main_input, doc_input], outputs=main_output)

	model.compile(loss='binary_crossentropy', optimizer='adam')


	#doc_encoder = Model(inputs=doc_input, outputs=x_doc_mult)
	#doc_encoder.compile(loss='mse', optimizer='adam')

	return model, None#doc_encoder


def initialize_network_C_deep(input_shape, width=256, nb_experts=2, dropout=0.5):
	# this is the basic version of the meta-lr

	main_input = Input(shape=input_shape, name='main_input')

	doc_input = Input(shape=input_shape, name='doc_input')

	x_doc_mult = Dense(width, activation='selu')(doc_input)
	x_doc_mult = Dropout(dropout)(x_doc_mult)
	x_doc_mult = Dense(width//2, activation='selu')(x_doc_mult)
	#x_doc_mult = Dropout(dropout_2)(x_doc_mult)
	x_doc_mult = Dense(nb_experts, activation='softmax')(x_doc_mult)#, kernel_regularizer=regularizers.l1_l2(l1=regularization[0], l2=regularization[1]))(x_doc_mult)


	x_sent = Dense(width, activation='selu')(main_input)
	x_sent = Dropout(dropout)(x_sent)
	x_sent = Dense(width//2, activation='selu')(x_sent)
	#x_sent = Dropout(dropout_2)(x_sent)
	x_sent = Dense(nb_experts, activation='sigmoid')(x_sent)

	
	main_output = Dot(axes=1)([x_sent, x_doc_mult])
	#main_output = Activation('sigmoid')(main_output)


	#main_output = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=regularization[0], l2=regularization[1]))(x)

	model = Model(inputs=[main_input, doc_input], outputs=main_output)

	model.compile(loss='binary_crossentropy', optimizer='adam')


	doc_encoder = Model(inputs=doc_input, outputs=x_doc_mult)
	doc_encoder.compile(loss='mse', optimizer='adam')

	return model, doc_encoder




