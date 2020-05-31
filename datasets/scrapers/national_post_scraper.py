#from urllib.request import Request
#import urllib.request
import requests
from bs4 import BeautifulSoup
import textblob
import time
import gzip

from .utils import url_to_filename, clean_text, get_sentences, pair_pq_with_sentences, write_pq_to_file, describe_integer_list

def get_paragraph_texts(soup):
	quote_items = soup.find_all("blockquote")
	# get the visible text of the article
	paragraph_items = [] #soup.find_all("p")
	for p in soup.find_all("p"):
		has_duplicate = False
		for q in quote_items:
			if p in q:
				quote_items.remove(q)
				has_duplicate = True
				break
		if not has_duplicate:
			paragraph_items.append(p)
	# find valid paragraph items (experimentally determined -- may be missing some) 
	paragraph_texts = [c.text for c in paragraph_items \
		if str(c).startswith("<p>") 
		or str(c).startswith('<p lang="en') 
		or str(c).startswith('<p class="western"') 
		or str(c).startswith('<p class="p') 
		or str(c).startswith('<p class="P') 
		or str(c).startswith('<p class="Standard"')
		or str(c).startswith('<p class="gmail-p1"')
		or str(c).startswith('<p class="x_ydpce29fb77MsoNormal"')
		or str(c).startswith('<p class="x_MsoNormal"')
		or str(c).startswith('<p style="font-weight:')
		or str(c).startswith('<p class="story-body-text story-content">')
		or str(c).startswith('<p class="m_1392363582732395558x_MsoNormal">')	
	]

	paragraph_texts = [clean_text(s) for s in paragraph_texts]
	return paragraph_texts


def get_soup(url_str):

	worked = None
	err_msg = ""
	soup = None


	try:
		r = requests.get(url_str)
		mystr = r.text

		worked = True


	except Exception as e:
		print("FAILED:", url_str)
		print(e)
		err_msg = "Failed urllib request"
		worked = False


	if worked:
		try:
			soup = BeautifulSoup(mystr, "lxml")
		except:
			err_msg = "BeautifulSoup error"
			worked = False

	return worked, soup, err_msg

def is_english(soup):
	# determine the language
	main_item = str(soup.find_all("html")[0])
	lang_index = main_item.find(" lang=")

	return 'lang="en' in main_item[lang_index:100]

def get_pull_quote_texts(soup):

	# get the *real* pull quotes
	# <blockquote class="pn_pullquote">
	#	https://nationalpost.com/opinion/munk-debates-capitalism-is-key-in-the-fight-against-climate-change
	#	https://nationalpost.com/opinion/munk-debates-ending-climate-change-requires-the-end-of-capitalism-as-we-know-it
	quote_items = soup.find_all("blockquote")
	pull_quote_texts = []
	for q in quote_items:
		q_str = str(q)
		if q_str.startswith('<blockquote class="pn_pullquote">'):
			text = q.text
			# remove the <div> ... </div>
			div_items = q.find_all("div")
			if len(div_items) > 0:
				text = text.replace(div_items[0].text, "")
			text = clean_text(text)
			text = text.replace("\n", "").strip()
			if len(text) > 0:
				pull_quote_texts.append(text)

	return pull_quote_texts



def scrape(url_list, save_dir, err_logger, skip_logger, skippable_urls):

	ALL_CHARS = set()

	total_pqs = 0

	all_p_samples = []


	auto_skipped = []
	for i_url, url_str in enumerate(url_list):
		# noticed that beyond about June 19, 2017, there are no more PQs
		if i_url > 28000: break # and not (i_url >= 21589 and i_url <= 24121): continue
		if url_str in skippable_urls:
			auto_skipped.append(i_url)
			continue

		if len(auto_skipped) > 0:
			print("\033[1;34mAUTO SKIPPED: {}\033[0m".format(describe_integer_list(auto_skipped)))
			auto_skipped = []


		print("\n\033[1;106mURL\033[0m {} ({:.4f}%, {})\t{}".format(i_url, 100*(i_url+1)/len(url_list), total_pqs, url_str))

		worked, soup, err_msg = get_soup(url_str)

		if not worked:
			err_logger("{}\t{}\t{}".format(i_url, err_msg, url_str))
			skip_logger("{}\t{}".format("REQ FAIL", url_str))
			continue

		if not is_english(soup):
			print("\tSKIPPING... NOT ENGLISH")
			skip_logger("{}\t{}".format("NON ENGLISH", url_str))
			continue



		pull_quote_texts = get_pull_quote_texts(soup)


		if len(pull_quote_texts) == 0:
			print("\tSKIPPING... NO PULL QUOTES")
			skip_logger("{}\t{}".format("NO PQ", url_str))
			continue


		paragraph_texts = get_paragraph_texts(soup)


		paragraph_texts = [p for i_p, p in enumerate(paragraph_texts) if not (
				p.startswith('Last Updated')
				or (i_p == 0 and (" EST" in p or "EDT" in p))
				or p.startswith("We want to improve your reading experience")
				or p.startswith("• Email")
				or p.startswith("Subscribe to our news podcast")
			)]
		
		#April 8, 202011:52 PM EDT

		all_sentences = get_sentences(paragraph_texts)

		nb_sentences = len(all_sentences)

		if nb_sentences <= 5:
			print("\tSKIPPING... TOO FEW SENTENCES")
			err_logger("{}\tToo few sentences ({})\t{}".format(i_url, nb_sentences, url_str))
			skip_logger("{}\t{}".format("FEW SENTS", url_str))
			continue

		# for later inspection... let's look at the set of characters used
		ALL_CHARS.update(set(''.join(all_sentences)))



		# pair PQs with source sentences and write to file
		valid_pq_count = 0
		
		edited_pq_texts = []
		source_pq_texts = []

		for q in pull_quote_texts:
			results, worked = pair_pq_with_sentences(q, all_sentences)
			if worked:
				edited_pq_texts.append(get_sentences([q]))
				source_pq_texts.append(results)
				total_pqs += 1
				valid_pq_count += 1
			else:
				err_logger("{}\tToo low of word coverage ({}): {}".format(i_url, q[:30], url_str))

		
		if valid_pq_count > 0:
			write_pq_to_file(i_url, url_str, all_sentences, edited_pq_texts, source_pq_texts, save_dir)
			skip_logger("{}\t{}".format("DONE", url_str))
		else:
			skip_logger("{}\t{}".format("NO VALID PQ", url_str))
