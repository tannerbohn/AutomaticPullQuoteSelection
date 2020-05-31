import sys, os
CODE_FILE_LOC = "/".join(os.path.realpath(__file__).split("/")[:-1])
sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, CODE_FILE_LOC)

import settings

import pickle
import glob
import unicodedata
import difflib



ALL_CHARS = set()

def clean_text(text):
	'''
	é 233 True
	— 8212 True
	‘ 8216 True
	’ 8217 True
	“ 8220 True
	” 8221 True
	… 8230 True
	'''

	cleaned_text = text
	#cleaned_text = cleaned_text.replace('é', 'e')
	cleaned_text = cleaned_text.replace('—', ' - ')

	#cleaned_text = cleaned_text.replace('‘', '\'')
	#cleaned_text = cleaned_text.replace('’', '\'')
	#cleaned_text = cleaned_text.replace('“', '\"')
	#cleaned_text = cleaned_text.replace('”', '\"')

	return cleaned_text


def split_punct(text):
	text = text.replace('“', '"')
	text = text.replace('”', '"')

	for ch in '\'’':
		for s in ["s", "re", "m", "ll", "ve", "d", "nt", "t"]:
			text = text.replace('{}{}'.format(ch, s), s)

	for ch in ',.!?"();:':
		text = text.replace(ch, " "+ch+" ")
	words = text.split()
	return words


def compute_word_inclusions(edited_pq, pq_source):
	# TODO: deal with punctuation? if at the end/start of a word, it can mess up matches
	edited_pq_words = split_punct(edited_pq)
	pq_source_words = split_punct(pq_source)
	S1 = split_punct(edited_pq.lower())
	S2 = split_punct(pq_source.lower())
	matches = difflib.SequenceMatcher(None, S1, S2).get_matching_blocks()
	match_info = []
	if matches[0].b > 0:
		match_info.append((pq_source_words[0:matches[0].b], 0))
	for i_m, match in enumerate(matches):
		# get stuff in match
		if matches[i_m].size > 0:
			match_info.append((pq_source_words[matches[i_m].b:matches[i_m].b+matches[i_m].size], 1))
		# get stuff after match if we're not at the end yet
		if i_m < len(matches)-1:
			if matches[i_m].b+match.size < matches[i_m+1].b:
				match_info.append((pq_source_words[matches[i_m].b+match.size:matches[i_m+1].b], 0))
		elif matches[i_m].b+match.size < len(pq_source_words):
			if matches[i_m].b+match.size < len(pq_source_words):
				match_info.append((pq_source_words[matches[i_m].b+match.size:len(pq_source_words)], 0))
	expanded_match_info = []
	for item in match_info:
		included = item[1]
		expanded_match_info.extend([(w, included) for w in item[0]])
	return expanded_match_info


def parse_pq_file(text):
	global ALL_CHARS
	# TODO: examine set of all characters, clean up

	'''
	return dict of form:
	{
		"url": "...",
		"sentences": ["", "", ..],
		"inclusions":[0, 0, 1, 0, 0, 2, 2, ...],
		"edited_pqs": ["", "", ..]

	}

	'''

	char_set = set(text)
	ALL_CHARS.update(char_set)

	all_sec_data = dict()
	all_sec_data['url'] = None
	all_sec_data['sentences'] = []
	all_sec_data['inclusions'] = []
	all_sec_data['edited_pqs'] = []

	# pre_body_text contains: url and edited pq/source pq pairings
	#	- this can have multiple sections (one for each pq)
	# body text contains the article body
	pre_body_text, body_text = text.split("========")

	sections = pre_body_text.split("----\n")

	all_sec_data['sentences'] = [s.strip() for s in body_text.strip().split('\n')]

	# initialize inclusions to 0 (not in any pq)
	all_sec_data['inclusions'] = [0 for _ in all_sec_data['sentences']]

	pq_index = 1
	for i_sec, sec_text in enumerate(sections):
		
		if i_sec == 0:
			url = sec_text.split("\n")[0].strip()
			all_sec_data['url'] = url
			continue

		sec_data = dict()

		try:
			edited_pq_text, source_pq_text = sec_text.strip().split("\n\n")
		except:
			print(pre_body_text)
			continue

		all_sec_data['edited_pqs'].append([l[3:] for l in edited_pq_text.strip().split("\n")])

		source_sentences = [l[3:] for l in source_pq_text.strip().split("\n")]
		
		inclusions = []
		clean_sentences = []

		included_sentences =[]

		for s in source_sentences:
			try:
				all_sec_data['inclusions'][all_sec_data['sentences'].index(s)] = pq_index
			except:
				print(sec_text)
				exit()
		pq_index += 1

		#word_inclusions = compute_word_inclusions(edited_pq_text, ' '.join(included_sentences))
		#sec_data['word_inclusions'] = word_inclusions

	all_sec_data['nb_pq'] = pq_index-1

	return all_sec_data

def preprocess_pull_quotes(directories):

	all_data = []

	for directory in directories:
	
		file_list = glob.glob("{}/*.txt".format(directory.rstrip("/")))
		#print(file_list)

		for fname in file_list:

			with open(fname, "r") as f:
				text = f.read()
				text = clean_text(text)

				all_data.append(parse_pq_file(text))

	return all_data


if __name__ == "__main__":

	pull_quotes = preprocess_pull_quotes(settings.PQ_SAMPLES_DIRS)


	#for ch in sorted(list(ALL_CHARS)):
	#	print(ch, ord(ch), ord(ch) > 128)
