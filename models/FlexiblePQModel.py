
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

	def __init__(self, sent_encoder, doc_normalize=False, mode="v1"):

		self.name = "flexible"
		self.sent_encoder = sent_encoder

		self.doc_normalize = doc_normalize

		self.mode=mode


		return

	def reset(self):
		self.sent_encoder.reset()
		return

	def set_mode(self, new_mode):

		self.mode = new_mode


	def encode(self, X):
		
		X_features = []

		for sentences, context in X:

			query_encs = []
			for s in sentences:
				query_encs.append(self.sent_encoder.encode(s))

			query_enc_avg = np.average(query_encs, axis=0)

			if not self.use_context:
				X_features.append(query_enc_avg)

			else:
				
				context_encs = []
				for s in context:
					context_encs.append(self.sent_encoder.encode(s))

				context_enc_avg = np.average(context_encs, axis=0)
				

				diff_vec = query_enc_avg - context_enc_avg
				#if self.vec_combination == "average":
				X_features.append(np.array(list(query_enc_avg)+list(diff_vec)))   #list(query_enc_avgdiff_vec)
				
		


		return np.array(X_features)
		



	def prepare_data(self, articles, neg_radius=3):

		# for each article, compute the document embedding and then store the learned paramaters
		# next, train a model to predict parameters given the embedding (return a list of embeddings
		#	maybe so that they can be clustered?)

		K.clear_session()

		#neg_radius = 3

		X_docs = []
		doc_encs = []
		y = []

		for i_article, article in enumerate(articles):
			if i_article % 500 == 0: print("\t{:.2f}".format(100*i_article/len(articles)))

			doc_enc = self.sent_encoder.encode(article['sentences']).astype(np.float32)

			sentences = []
			labels = []
			if neg_radius == None:
				sentences = article['sentences']
				labels = article['inclusions']
			else:
				
				for i in range(len(article['sentences'])):
					start_index = max(0, i-neg_radius)
					end_index = i+neg_radius
					if any(article['inclusions'][start_index:end_index+1]):
						sentences.append(article['sentences'][i])
						labels.append(article['inclusions'][i])
			

			X_doc = np.array([self.sent_encoder.encode(s) for s in sentences]).astype(np.float32)
			y_doc = [1 if v >= 1 else 0 for v in labels]


			X_docs.extend(X_doc)
			doc_encs.extend([doc_enc for _ in X_doc])
			y.extend(y_doc)


		self.X_docs = np.array(X_docs).astype(np.float32)
		self.doc_encs = np.array(doc_encs).astype(np.float32)
		self.y = np.array(y)


		

	def train_model(self, width=128, epochs=10, batch_size=128, verbose=1, regularization=(0,0), pos_weight=1, activation='tanh', nb_experts=2, dropout=0.25):

		K.clear_session()

		callback = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

		val_split = 0.2

		if self.mode == "v1":
			# logistic regression
			self.model, self.doc_encoder = initialize_network_v1(input_shape=self.X_docs[0].shape, regularization=regularization)
			self.model.fit(self.X_docs, self.y, epochs=epochs, batch_size=batch_size, verbose=verbose, class_weight={0: 1, 1: pos_weight}, validation_split=val_split, callbacks=[callback])

		elif self.mode == "v2":
			# lr with doc input as well
			self.model, self.doc_encoder = initialize_network_v2(input_shape=self.X_docs[0].shape, regularization=regularization)
			self.model.fit([self.X_docs, self.doc_encs], self.y, epochs=epochs, batch_size=batch_size, verbose=verbose, class_weight={0: 1, 1: pos_weight}, validation_split=val_split, callbacks=[callback])

		elif self.mode == "v3":
			# basic meta-lr
			self.model, self.doc_encoder = initialize_network_v3(input_shape=self.X_docs[0].shape, nb_experts=nb_experts, regularization=regularization)
			self.model.fit([self.X_docs, self.doc_encs], self.y, epochs=epochs, batch_size=batch_size, verbose=verbose, class_weight={0: 1, 1: pos_weight}, validation_split=val_split, callbacks=[callback])


		elif self.mode == "v1_deep":
			# logistic regression
			self.model, self.doc_encoder = initialize_network_v1_deep(input_shape=self.X_docs[0].shape, width=width, regularization=regularization)
			self.model.fit(self.X_docs, self.y, epochs=epochs, batch_size=batch_size, verbose=verbose, class_weight={0: 1, 1: pos_weight}, validation_split=val_split, callbacks=[callback])

		elif self.mode == "v2_deep":
			# lr with doc input as well
			self.model, self.doc_encoder = initialize_network_v2_deep(input_shape=self.X_docs[0].shape, width=width, regularization=regularization)
			self.model.fit([self.X_docs, self.doc_encs], self.y, epochs=epochs, batch_size=batch_size, verbose=verbose, class_weight={0: 1, 1: pos_weight}, validation_split=val_split, callbacks=[callback])

		elif self.mode == "v3_deep":
			# basic meta-lr
			self.model, self.doc_encoder = initialize_network_v3_deep(input_shape=self.X_docs[0].shape, width=width, nb_experts=nb_experts, dropout=dropout)
			self.model.fit([self.X_docs, self.doc_encs], self.y, epochs=epochs, batch_size=batch_size, verbose=verbose, class_weight={0: 1, 1: pos_weight}, validation_split=val_split, callbacks=[callback])

		#pos_frac = np.average(self.y)
		#print("inv pos frac:", 1/pos_frac)
		#pos_weight = pos_weight/pos_frac
		#print("pos weight:", pos_weight)

		#self.model.fit([self.X_docs, self.doc_encs], self.y, epochs=epochs, batch_size=batch_size, verbose=verbose, class_weight={0: 1, 1: pos_weight})

		#return self.doc_encoder.predict(self.doc_encs)	


	def predict_article(self, sentences, document=None):

		doc_enc = self.sent_encoder.encode(document).astype(np.float32)

		X_doc = np.array([self.sent_encoder.encode(s, document=document) for s in sentences]).astype(np.float32)


		if "v2" in self.mode or "v3" in self.mode:
			y_pred = self.model.predict([X_doc, np.array([doc_enc for _ in X_doc]).astype(np.float32)])
		else:
			y_pred = self.model.predict(X_doc)

		y_pred = y_pred[:,0]

		return y_pred



###############################################################################

def initialize_network_v1(input_shape, regularization=(0,0)):
	# V1 essentially implements logistic regression with only the candidate text as input

	main_input = Input(shape=input_shape, name='main_input')

	main_output = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=regularization[0], l2=regularization[1]))(main_input)

	model = Model(inputs=main_input, outputs=main_output)

	model.compile(loss='binary_crossentropy', optimizer='adam')

	return model, None

def initialize_network_v1_deep(input_shape, width=256, regularization=(0,0)):
	# V1 essentially implements logistic regression with only the candidate text as input

	main_input = Input(shape=input_shape, name='main_input')

	x = Dense(width, activation='selu')(main_input)
	x = Dropout(0.25)(x)
	x = Dense(width//2, activation='selu')(x)

	main_output = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=regularization[0], l2=regularization[1]))(x)

	model = Model(inputs=main_input, outputs=main_output)

	model.compile(loss='binary_crossentropy', optimizer='adam')

	return model, None

###############################################################################

def initialize_network_v2(input_shape, regularization=(0,0)):
	# this version simply concatenates the candidate and doc encodigs for logistic regression

	main_input = Input(shape=input_shape, name='main_input')

	doc_input = Input(shape=input_shape, name='doc_input')

	
	x = Concatenate()([main_input, doc_input])

	main_output = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=regularization[0], l2=regularization[1]))(x)

	model = Model(inputs=[main_input, doc_input], outputs=main_output)

	model.compile(loss='binary_crossentropy', optimizer='adam')


	return model, None

def initialize_network_v2_deep(input_shape, width=256, regularization=(0,0)):
	# this version simply concatenates the candidate and doc encodigs for logistic regression

	main_input = Input(shape=input_shape, name='main_input')

	doc_input = Input(shape=input_shape, name='doc_input')

	
	x = Concatenate()([main_input, doc_input])

	x = Dense(width, activation='selu')(x)
	x = Dropout(0.25)(x)
	x = Dense(width//2, activation='selu')(x)

	main_output = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=regularization[0], l2=regularization[1]))(x)

	model = Model(inputs=[main_input, doc_input], outputs=main_output)

	model.compile(loss='binary_crossentropy', optimizer='adam')


	return model, None

###############################################################################


def initialize_network_v3(input_shape, nb_experts=2, regularization=(0,0)):
	# this is the basic version of the meta-lr

	main_input = Input(shape=input_shape, name='main_input')

	doc_input = Input(shape=input_shape, name='doc_input')

	x_doc_mult = Dense(nb_experts, activation='softmax', activity_regularizer=regularizers.l1_l2(l1=regularization[0], l2=regularization[1]))(doc_input) #, kernel_regularizer=regularizers.l1_l2(l1=regularization[0], l2=regularization[1]))(doc_input)

	sent_enc = Dense(nb_experts, activation='linear')(main_input)

	main_output = Dot(axes=1)([sent_enc, x_doc_mult])
	main_output = Activation('sigmoid')(main_output)

	model = Model(inputs=[main_input, doc_input], outputs=main_output)

	model.compile(loss='binary_crossentropy', optimizer='adam')


	#doc_encoder = Model(inputs=doc_input, outputs=x_doc_mult)
	#doc_encoder.compile(loss='mse', optimizer='adam')

	return model, None#doc_encoder


def initialize_network_v3_deep(input_shape, width=256, nb_experts=2, dropout=0.25):
	# this is the basic version of the meta-lr

	main_input = Input(shape=input_shape, name='main_input')

	doc_input = Input(shape=input_shape, name='doc_input')

	x_doc_mult = Dense(width, activation='selu')(doc_input)
	x_doc_mult = Dropout(dropout)(x_doc_mult)
	x_doc_mult = Dense(width//2, activation='selu')(x_doc_mult)
	x_doc_mult = Dropout(dropout/2)(x_doc_mult)
	x_doc_mult = Dense(nb_experts, activation='softmax')(x_doc_mult)#, kernel_regularizer=regularizers.l1_l2(l1=regularization[0], l2=regularization[1]))(x_doc_mult)

	x_sent = Dense(width, activation='selu')(main_input)
	x_sent = Dropout(dropout)(x_sent)
	x_sent = Dense(width//2, activation='selu')(x_sent)
	x_sent = Dropout(dropout/2)(x_sent)
	x_sent = Dense(nb_experts, activation='linear')(x_sent)

	
	main_output = Dot(axes=1)([x_sent, x_doc_mult])
	main_output = Activation('sigmoid')(main_output)


	#main_output = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=regularization[0], l2=regularization[1]))(x)

	model = Model(inputs=[main_input, doc_input], outputs=main_output)

	model.compile(loss='binary_crossentropy', optimizer='adam')


	doc_encoder = Model(inputs=doc_input, outputs=x_doc_mult)
	doc_encoder.compile(loss='mse', optimizer='adam')

	return model, doc_encoder




