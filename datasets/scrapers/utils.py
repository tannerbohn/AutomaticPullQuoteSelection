import os
from urllib.request import Request
import urllib.request
from bs4 import BeautifulSoup
import textblob
import time
from difflib import SequenceMatcher
import random
import glob
import gzip
import copy


def longestSubstringFinder(string1, string2):
	answer = ""
	len1, len2 = len(string1), len(string2)
	for i in range(len1):
		for j in range(len2):
			lcs_temp=0
			match=''
			while ((i+lcs_temp < len1) and (j+lcs_temp<len2) and string1[i+lcs_temp] == string2[j+lcs_temp]):
				match += string2[j+lcs_temp]
				lcs_temp+=1
			if (len(match) > len(answer)):
				answer = match
	return answer


def remove_nonalphanum(text):

	return ''.join([ch for ch in text if ch in " qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890"])


def pair_pq_with_sentences(pq_text, sentences, w=2):
	# INPUTS:
	# 	pq_text: the edited pull quote text
	# 	sentence: the sentences from the article
	#	w: how many sentences before and after to include in final context
	# OUTPUTS:
	# 	context (as list of strings with @@ where appropriate)

	# https://misc.flogisoft.com/bash/tip_colors_and_formatting

	print("-----------------------")
	print("\033[7mEDITED PQ\033[0m: {}".format(pq_text))


	pq_text_clean = remove_nonalphanum(pq_text).lower()
	sentences_clean = [remove_nonalphanum(s).lower() for s in sentences]

	pq_words = set(pq_text_clean.split())
	candidates = []
	for i_s, s in enumerate(sentences_clean):
		#print("\n\nanalysing:", s)
		longest_substring = longestSubstringFinder(pq_text_clean, s)
		common_len = len(longest_substring)
		nb_words = len(longest_substring.split())
		len_frac_1 = 0 if len(pq_text_clean) == 0 else len(longest_substring)/len(pq_text_clean)
		len_frac_2 = 0 if len(s) == 0 else len(longest_substring)/len(s)
		#print("\t", longest_substring, common_len, nb_words, len_frac_1, len_frac_2)
		candidates.append((i_s, s, len_frac_1, len_frac_2))#
	
	# sort candidates from low to high
	candidates = sorted(candidates, key=lambda k: k[2]*len(pq_text_clean)**0.5+k[3]*len(sentences_clean[k[0]])**0.5)
	chosen = []

	while True:
		# get the next most promising candidate
		c = candidates.pop()

		# get the set of words in the candidate sentence
		c_words = set(c[1].split())

		# how many of the candidate words intersection pull quote words (that have not been covered already)?
		new_words = c_words.intersection(pq_words)
		nb_new_words = len(new_words)
		
		'''
		add the candidate if...
			it contains at least 1 new word
				AND
			either the number of new words >= 4 or the intersection length is > 1/4 of the candidate length
				AND
			either no candidates have been chosen yet OR the candidate is no more than 4 sentences away from a previously chosen candidate
		'''
		if nb_new_words >= 1 and (nb_new_words >= 4 or c[3] > 0.25) and (len(chosen)==0 or min([abs(c[0] - c2[0]) for c2 in chosen]) <= 4):
			chosen.append(c)
			
			if nb_new_words < 3:
				c_show_str = " ".join([w if w not in new_words else "\033[1;33m"+w+"\033[0m" for w in c[1].split()])
			else:
				c_show_str = " ".join([w if w not in new_words else "\033[1;32m"+w+"\033[0m" for w in c[1].split()])
		
			print("\tADDED [{}%]: {}".format(int(100*c[3]), c_show_str))
			pq_words = pq_words - c_words

		if len(candidates) == 0 or nb_new_words == 0: break

	if len(chosen) == 0: return [], False	

	# sort chosen source sentences by location
	chosen = sorted(chosen, key=lambda k: k[0])

	l_sum = sum([len(c[1]) for c in chosen])
	print()
	print("\tLENGTH COVERAGE: {:.1f}%".format(100*l_sum/len(pq_text)))
	word_coverage = 1 - len(pq_words)/len(set(pq_text.split()))
	if word_coverage >= 0.95:
		colour_str = "\033[1;44m"
	elif word_coverage >= 0.85:
		colour_str = "\033[1;42m"
	elif word_coverage >= 0.7:
		colour_str = "\033[1;43m"
	elif word_coverage >= 0.6:
		colour_str = "\033[1;41m"
	else:
		colour_str = "\033[4m\033[1;31m"
	
	print("\tWORD COVERAGE: {}{:.1f}%\033[0m".format(colour_str, 100*word_coverage))
	
	#print("CHOSEN:", chosen)

	chosen_indices = [c[0] for c in chosen]
	min_index = min(chosen_indices)
	max_index = max(chosen_indices)
	indexed_sentences = [(i_s, s) for i_s, s,  in enumerate(sentences)]
	
	#context_start_index = max(0, min_index-w)
	#context_end_index = min(len(sentences)-1, max_index+w)

	results = []
	#for i_s, s in indexed_sentences[context_start_index:context_end_index+1]:
	#if i_s in chosen_indices:
	#	results.append("@@ "+sentences[i_s])
	for index, _, _, _ in chosen:
		results.append(sentences[index])

	return results, word_coverage >= 0.6


def url_to_filename(save_dir, url_str):

	filename_core = url_str.split(".com/")[1]
	filename_core = filename_core.rstrip("/").replace("/", "-")
	filename_core = filename_core[:50]

	
	copy_num = 0

	filename = "{}/{}_{}.txt".format(save_dir.rstrip('/'), filename_core, copy_num)

	while os.path.exists(filename):
		copy_num += 1
		filename = "{}/{}_{}.txt".format(save_dir.rstrip('/'), filename_core, copy_num)

	return filename


def clean_text(text):
	text = text.replace("\xa0", " ") # non-breaking space
	text = text.replace("\xad", "") # soft hyphens
	text = text.replace("\u200b", "") # zero length whitespace (whyyy)
	return text



def get_sentences(paragraph_texts):
	# for each of the content items, split it into sentences, them merge all sentence lists
	all_sentences_raw = []
	for t in paragraph_texts:
		sents = [str(s) for s in textblob.TextBlob(t).sentences]
		all_sentences_raw.extend(sents)

	'''
	fix some issues with sentence splitting
		- use a newline char as a sentence boundary
		- recognize when a quotation ends a sentence
			- ." or ?" or !"
			- but only when the stuff after it is long enough to 
				probably be another sentence (ex. ...blabla." he said. <- "he said"
				is part of the same sentence) 
	'''
	all_sentences = []
	for raw_s in all_sentences_raw:
		if '\n' in raw_s:
			sents = raw_s.split("\n")
		elif '.”' in raw_s and raw_s.index('.”') < len(raw_s)-25:
			sents = raw_s.replace(".”", ".”@@").split("@@")
			sents = [s.strip() for s in sents if len(s.strip()) > 0]
			#print("\033[1;106m>>>>>\033[0m", sents)
		
		elif '?”' in raw_s and raw_s.index('?”') < len(raw_s)-25:
			sents = raw_s.replace("?”", "?”@@").split("@@")
			sents = [s.strip() for s in sents if len(s.strip()) > 0]
			#print("\033[1;106m>>>>>\033[0m", sents)
		
		elif '!”' in raw_s and raw_s.index('!”') < len(raw_s)-25:
			sents = raw_s.replace("!”", "!”@@").split("@@")
			sents = [s.strip() for s in sents if len(s.strip()) > 0]
			#print("\033[1;106m>>>>>\033[0m", sents)
		
		else:
			sents = [raw_s]
		all_sentences.extend(sents)

	all_sentences = [s.strip() for s in all_sentences if len(s.strip()) > 0]

	return all_sentences


def write_pq_to_file(i_url, url_str, all_sentences, edited_pq_texts, source_pq_texts, save_dir):

	


	lines = []
	lines.append("{}\n{}".format(url_str, i_url))

	for edited, source in zip(edited_pq_texts, source_pq_texts):
		lines.append("\n----\n")
		for l in edited:
			lines.append("E: "+l)

		lines.append("\n")

		for l in source:
			lines.append("S: "+l)

	lines.append("\n========\n")
	for s in all_sentences:
		lines.append(s)


	# make sure we have a unique filename
	filename = url_to_filename(save_dir, url_str)


	with open(filename, "w") as f:
		for l in lines:
			f.write(l+"\n")

	print("saved to file:", filename)

	return

def describe_integer_list(values):

	if len(values) == 1:
		return str(values[0])

	values = sorted(values)

	ranges = []

	new_group = []
	for i_v, v in enumerate(values):
		if i_v == 0:
			new_group.append(v)
		elif v == values[i_v-1]+1:
			new_group.append(v)
		else:
			ranges.append((min(new_group), max(new_group)))
			new_group = [v]

	ranges.append((min(new_group), max(new_group)))

	descr = ', '.join([str(m) if m == M else '{}-{}'.format(m, M) for m, M in ranges])

	return descr