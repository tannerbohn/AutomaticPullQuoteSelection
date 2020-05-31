import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

import random

import settings
from utils.pq_preprocessing import preprocess_pull_quotes
from models.sentence_encoders import HandcraftedEncoder


sns.set(rc={'figure.figsize':(4, 5)})
plt.style.use('fivethirtyeight')


def view_feature(index, name="variable", clip=None, bw='silverman'):
	global pq_encs, non_pq_encs
	ax = sns.kdeplot(non_pq_encs[:,index], label="non-PQ sentences", clip=clip, bw=bw, shade=True, shade_lowest=True)
	ax.lines[-1].set_linestyle("-")
	sns.kdeplot(pq_encs[:,index], label="PQ sentences", clip=clip, bw=bw)
	ax.lines[-1].set_linestyle("--")
	'''
	" ": empty
	"-": solid
	"--": dashed
	"-.": dash-dot
	":": dotted
	'''
	#ax.set_title('{} distribution for text types'.format(name))
	ax.set_ylabel('Density')
	ax.set_xlabel('{}'.format(name))
	plt.legend();
	plt.tight_layout()
	plt.savefig('results/generated_charts/{}.pdf'.format(name), format='pdf')
	#plt.show()
	plt.cla()



articles_data = preprocess_pull_quotes(directories=settings.PQ_SAMPLES_DIRS)
#random.shuffle(articles_data)
#articles_data = articles_data[:100]

sent_encoder = HandcraftedEncoder(features="all")

pq_encs = []
non_pq_encs = []

nb_samples = len(articles_data) #2000
print("Computing sentence features...")
for i_article, article in enumerate(articles_data[:nb_samples]):
	if i_article % 100 == 0:
		print("{:.2f}%".format(100*i_article/nb_samples))	
	for sentence, inclusion in zip(article['sentences'], article['inclusions']):
		enc = sent_encoder.encode(sentence, document=article['sentences'])
		if inclusion == 0:
			non_pq_encs.append(enc)
		else:
			pq_encs.append(enc)


print("# pull quotes = ", len(pq_encs))
print("# non_pull quotes = ", len(non_pq_encs))

pq_encs = np.array(pq_encs)
non_pq_encs = np.array(non_pq_encs)

feature_list = [
	("Quote_count", (-1, 6)),
	("Len_total", (-10, 400)),
	("Sent_position", None),
	("R_Flesch", (-50, 150)),
	("R_CLI", (-10, 30)),
	("R_difficult", (-5, 120)),
	("Len_word_avg", (0, 10)),
	("POS_CD", (-10, 20)),
	("POS_JJ", (-10, 30)),
	("POS_MD", (-10, 15)),
	("POS_NN", (-10, 75)),
	("POS_NNP", (-10, 60)),
	("POS_PRP", (-10, 50)),
	("POS_RB", (-10, 50)),
	("POS_VB", (-10, 30)),
	("A_pos", (-10, 60)),
	("A_neg", (-10, 60)),
	("A_compound", None),
	("A_valence", (3, 8)),
	("A_arousal", (2, 6)),
	("A_concreteness", None)
]



for i_f, (f_name, clip) in enumerate(feature_list):
	print("Generating figure for", f_name)
	view_feature(index=i_f, name=f_name, clip=clip)
