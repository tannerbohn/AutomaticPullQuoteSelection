import numpy as np
import time
import random
import itertools
import argparse
import math

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Concatenate, RepeatVector, Subtract, Multiply, Dropout, Dot, Activation, Add
from keras.activations import sigmoid
import keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks

from sklearn.linear_model import LogisticRegression

from models.sentence_encoders import SentBERTEncoder
from models.SimplePQModel import SimplePQModel

import settings

from utils.pq_preprocessing import preprocess_pull_quotes
from utils.data_utils import get_article_partition
from utils.Evaluator import Evaluator


class TrueCharPositionEncoder:

	def __init__(self, char_sets=['“', '”'], quantiles=10):

		self.name = "true_char_position"
		self.char_sets = char_sets

		self.quantiles = quantiles

		self.shape = (quantiles*len(self.char_sets),)

	def encode(self, sentence, document=None):

		#text.count('“')+text.count('”')
		occurrence_vec = np.zeros((len(self.char_sets), self.quantiles))

		n = len(sentence)



		for i_ch, ch in enumerate(sentence):
			true_frac = 0 if n <= 1 else i_ch/(n-1)
			bucket = int(min(math.floor(true_frac * self.quantiles), self.quantiles-1))
			#print(bucket)

			for i_set, set_chars in enumerate(self.char_sets):
				if ch in set_chars:
					#print("\tyes")
					occurrence_vec[i_set][bucket] += 1
			#else:
			#	occurrence_vec[bucket] = 0


		return occurrence_vec.flatten()

class PredictedCharPositionEncoder:

	def __init__(self, sent_encoder, quantiles=10, layer_sizes=(), layer_dropouts=(), activation='selu'):

		self.name = "predicted_char_distribution"

		self.sent_encoder = sent_encoder

		self.quantiles = quantiles

		self.target_encoder = TrueCharPositionEncoder(quantiles=self.quantiles)

		self.layer_sizes = layer_sizes
		self.layer_dropouts = layer_dropouts
		self.activation = activation


		

	def fit(self, train_articles, val_articles, verbose=1, max_epochs=100):
		K.clear_session()

		X_train = []
		y_train = []

		X_val = []
		y_val = []

		split_labels = ['train' for _ in train_articles]+['val' for _ in val_articles]

		for article, label in zip(train_articles+val_articles, split_labels):

			sentences = article['sentences']

			sent_encs = np.array([self.sent_encoder.encode(s) for s in sentences])
			targets = [self.target_encoder.encode(s) for s in sentences]

			if label == 'train':
				X_train.extend(sent_encs)
				y_train.extend(targets)
			else:
				X_val.extend(sent_encs)
				y_val.extend(targets)



		X_train = np.array(X_train).astype(np.float32)
		y_train = np.array(y_train).astype(np.int32)

		X_val = np.array(X_val).astype(np.float32)
		y_val = np.array(y_val).astype(np.int32)


		callback = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

		self.model = self.get_model(X_train[0].shape, self.quantiles, self.layer_sizes, self.layer_dropouts, self.activation)
		
		self.model.fit(X_train, y_train, epochs=max_epochs, batch_size=128, verbose=verbose, validation_data=(X_val, y_val), callbacks=[callback])

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

	def get_model(self, input_shape, output_dim, layer_sizes=(), layer_dropouts=(), activation='selu'):
		
		main_input = Input(shape=input_shape, name='main_input')

		x = main_input

		for i_l, (width, dropout) in enumerate(zip(layer_sizes, layer_dropouts)):
			x = Dense(width, activation=activation)(x)
			x = Dropout(dropout)(x)


		main_output = Dense(output_dim, activation='sigmoid')(x)

		model = Model(inputs=main_input, outputs=main_output)

		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

		return model

class SimpleNNModel:

	def __init__(self, sent_encoder, layer_sizes=(), layer_dropouts=()):

		self.name = "simple_nn"

		self.sent_encoder = sent_encoder

		self.layer_sizes = layer_sizes
		self.layer_dropouts = layer_dropouts

	def fit(self, train_articles, val_articles, max_epochs=100, verbose=0):

		K.clear_session()

		X_train = []
		y_train = []

		X_val = []
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

			else:

				X_val.extend(sent_encs)
			
				y_val.extend(labels)


		X_train = np.array(X_train).astype(np.float32)
		y_train = np.array(y_train).astype(np.int32)


		#gates_train = (gates_train.T / np.sum(gates_train, axis=1)).T

		X_val = np.array(X_val).astype(np.float32)
		y_val = np.array(y_val).astype(np.int32)

		self.model = self.get_model()

		callback = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

		self.model.fit(X_train, y_train, epochs=max_epochs, batch_size=128, verbose=verbose, validation_data=(X_val, y_val), callbacks=[callback])



	def get_model(self):

		input_shape = self.sent_encoder.shape

		main_input = Input(shape=input_shape, name='main_input')

		x = main_input

		for width, dropout in zip(self.layer_sizes, self.layer_dropouts):

			x = Dense(width, activation='selu')(x)
			x = Dropout(dropout)(x)


		main_output = Dense(1, activation='sigmoid')(x)
		#main_output = Dense(1, activation='sigmoid')(merged)
		#main_output = Activation('sigmoid')(main_output)

		model = Model(inputs=main_input, outputs=main_output)

		model.compile(loss='binary_crossentropy', optimizer='adam')


		#doc_encoder = Model(inputs=doc_input, outputs=x_doc_mult)
		#doc_encoder.compile(loss='mse', optimizer='adam')

		return model

	def predict_article(self, sentences, document=None):

		sent_encs = np.array([self.sent_encoder.encode(s) for s in sentences]).astype(np.float32)


		y_pred = self.model.predict(sent_encs)
		
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

'''
pos_quote_occurrence = []
neg_quote_occurrence = []
for a in articles_data:
	for s, inc in zip(a['sentences'], a['inclusions']):
		has_quote = '“' in s or '”' in s
		if inc > 0:
			pos_quote_occurrence.append(has_quote)
		else:
			neg_quote_occurrence.append(has_quote)
'''




#pre_sent_encoder = SentBERTEncoder(precomputed_embeddings=settings.PRECOMPUTED_SBERT_EMBEDDINGS_FNAME)


timestamp = time.strftime("%F-%H:%M:%S")
results_file = open("results/TVD_{}.txt".format(timestamp), "w")


char_sets_options = [
	['“', '”', ',', '’', '-', '.', ' ', '?', '!']
]



#char_sets = ['Aa','Bb','Cc','Dd','Ee','Ff','Gg','Hh','Ii','Jj','Kk','Ll','Mm','Nn','Oo','Pp','Qq','Rr','Ss','Tt','Uu','Vv','Ww','Xx','Yy','Zz']

all_chars = [' ', 'e', 't', 'a', 'i', 'o', 'n', 's', 'r', 'h', 'l', 'd', 'c', 'u', 'm', 'p', 'f', 'g', 'w', 'y', 'b', ',', 'v', 'k', '’', '-', '“', '.', 'j', 'x', 
	'0', '”', '1', 'z', '2', 'q', "'", '9', '5', ':', '3', '(', '4', '6', '7', '8', ')', '$', '"', ';', '–', '?', '‘', '…', '[', ']', 'é', '/', '@', '!', '#', 'á', 
	'í', 'ó', '&', '_', '*', '•', 'ñ', '%', 'ú', 'ç', 'ü', '+', '~', 'è', '|', '⠀', 'ã', 'à', '️', 'ö', 'â', 'ï', 'ı', 'о', '\u2028', 'ô', 'а', '=', 'и', 'т', 'е', 
	'ğ', 'ò', '£', 'ê', '❤', 'н', 'р', 'с', 'л', 'î', 'ë', 'ş', 'в', '📷', 'я', '́', '©', '📸', '¿', '<', 'к', 'д', '\u200d', 'у', '✨', 'п', 'м', '🏼', '½', '\x80', 
	'¦', '♀', 'ь', 'г', '\u2063', 'з', '✌', '🔥', '🏽', 'õ', 'æ', 'б', '―', '¢', '″', '\u2060', '>', '😂', 'ы', '🤤', '・', '̇', 'й', 'û', '`', 'ø', '°', 'ч', '☀', 
	'🌴', '🌹', '§', 'č', '💜', '🏻', '🌜', '♎', '🌛', '€', '🔆', 'ш', '👍', '™', '🚀', 'š', '^', '💯', '😍', '🙌', 'х', '̈', 'ð', '′', '®', '¡', '𝐫', 'ä', 'ж', '💛', 'ł', 
	'🇸', '🌈', '💙', '\u200e', 'å', '🖤', '‐', '̂', '💥', '😉', 'щ', 'ё', '🤷', 'ц', '💖', '🍹', '🍊', '{', '}', '🤔', '💋', '\U0001f9e1', '\u200f', '𝐞', '🛑', 'º', '🇹', 
	'☕', 'ɪ', '🌊', '💁', '🇺', '🤗', '🙏', '×', '💚', '¯', '🌺', '→', '♂', '전', '율', '리', '✈', 'ń', '💦', '☁', '🌟', 'ʼ', '‚', '👏', '😏', '🤦', '🎉', '🎄', '💪', '👄', 
	'👑', '♥', '¹', 'ā', '💍', '𝐖', '𝐨', '𝐥', '👌', '🥂', '💅', '☝', '👋', '😃', '⚾', '😅', '🛁', '🇷', '̀', 'ю', 'э', '▪', 'ᴛ', 'ɴ', 'ᴀ', 'ᴇ', '🐝', '➡', '🌅', '⛱', 'ś', 
	'🇧', '💕', '⚡', '🌎', '\\', 'ツ', 'œ', '👯', '😭', '💔', '👀', '🍕', '🎶', '🙆', 'ž', '😊', '�', '📍', '🍃', '💎', '⛵', '♡', '😳', '\U0001f90d', '😐', 'ù', '🌸', '¬', '‑', 
	'👎', '\U0001f9d8', '−', '🐋', '🎀', '👸', '😆', '💸', '😪', '🍎', '👭', '😋', '🖕', '😑', '🐈', '👜', '🙀', '😼', '😽', '👊', '´', '🤓', '\U0001f929', '😎', '🤡', '🎅', '🔱', 
	'💄', '⚜', '🇫', '𝐝', '𝐓', '𝐚', '𝐯', '𝐌', '𝐦', '𝐢', '𝐭', '¾', 'ℓ', '¨', '👰', '😁', '🏋', '😝', '👇', '😚', '💏', '🗳', '🏙', '🕌', 'ф', '👗', '🛍', 'ʀ', 'ʟ', '💌', '🛣', 
	'🚖', '🔍', '🏬', '🌉', '🎭', '🏠', '🌳', '📏', 'ą', '🏅', '😩', '💀', '🎃', '👶', '￼', '🥞', 'ﬁ', '🐔', '🏕', '̊', '\U0001f974', '🙃', '✊', '̃', '🌚', '🌻', '😬', '🙋', '🙂', 
	'🇮', '☺', '🏄', '💐', '🏁', '🤠', '😢', '❗', '♐', '🍟', 'ｍ', 'ｕ', 'ｎ', 'ｒ', 'ｏ', 'ｅ', '🎼', 'ʔ', '≠', '😘', '😵', '🍅', '😮', '🐚', '🤸', '✮', '🦐', '🐠', '🌏', '⃣', '🛶', 
	'😻', '\U0001f9d0', '🇭', '🚜', 'ß', '💰', '\u2009', '¼', '💡', '🚁', '🇬', '🐣', '🗺', '\U0001f6f8', '🌑', '🤘', '🌋', '🥐', '🧀', '🍳', '🥓', '🍷', '🥑', '🍤', '🍸', '😰',
	 '─', '😜', '🌼', '🎠', '🙈', '🎡', '🏾', '🍑', '🍫', 'ć', '⛈', 'ʺ', '☾', '˚', '\U0001f976', '🏖', '\U0001f9da', '🐶', '🍁', '⚓', 'ż']


for char_sets in char_sets_options:
	for quantiles in [1, 2, 5, 10, 15]:
		sent_encoder = TrueCharPositionEncoder(char_sets = char_sets, quantiles=quantiles)
		model = SimplePQModel(sent_encoder=sent_encoder, clf_type=LogisticRegression, clf_args={'class_weight':'balanced', 'max_iter':1000, 'solver':'lbfgs'})
		#model = SimpleNNModel(sent_encoder=sent_encoder, layer_sizes=layer_sizes, layer_dropouts=layer_dropouts)
		model.fit(train_articles)
		#coefs = model.model.coef_[0]
		#coef_str = "\t".join([str(round(v, 3)) for v in coefs])
		#
		#val_accuracy = E.evaluate(model=model, articles=val_articles, verbose=0)
		test_accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)
		res_str = "{}\t{}\t{}\t{:.2f}".format(sent_encoder.name, char_sets, quantiles, 100*test_accuracy)
		print(res_str)
		results_file.write(res_str+"\n")
		results_file.flush()


'''
for quantiles in [2, 5, 10, 20]:
	sent_encoder = PredictedQuotePositionEncoder(sent_encoder=pre_sent_encoder, quantiles=quantiles, layer_sizes=(), layer_dropouts=())
	sent_encoder.fit(train_articles=train_articles, val_articles=val_articles, verbose=1, max_epochs=1 if quick_mode else 100)
	model = SimplePQModel(sent_encoder=sent_encoder, clf_type=LogisticRegression, clf_args={'class_weight':'balanced', 'max_iter':1000, 'solver':'lbfgs'})
	model.fit(train_articles)
	coefs = model.model.coef_[0]
	coef_str = "\t".join([str(round(v, 3)) for v in coefs])
	#
	#val_accuracy = E.evaluate(model=model, articles=val_articles, verbose=0)
	test_accuracy = E.evaluate(model=model, articles=test_articles, verbose=0)
	res_str = "{}\t{}\t{:.2f}\t{}".format(sent_encoder.name, quantiles, 100*test_accuracy, coef_str)
	print(res_str)
	results_file.write(res_str+"\n")
'''

#results_file.close()
#K.clear_session()