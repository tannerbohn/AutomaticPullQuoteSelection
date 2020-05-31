import os
import time
import argparse

from scrapers import national_post_scraper, cosmo_scraper, ottawa_citizen_scraper, intercept_scraper

def get_logger(file):
	def logger(message):
		file.write("\n"+message)
		file.flush()
	return logger

def get_url_list(filename, news_source):


	url_set = set()
	final_url_list = []

	with open(filename, "r") as f:
		lines = f.read().strip().split("\n")
		print("URL LIST {}: {} urls".format(filename, len(lines)))
		for url_str in lines:

			if news_source == "national-post":
				if '/opinion/' not in url_str and '/news/' not in url_str: continue
			elif news_source == "ottawa-citizen":
				if '/opinion/' not in url_str: continue

			if url_str not in url_set:
				final_url_list.append(url_str)
				url_set.add(url_str)

	return final_url_list


parser = argparse.ArgumentParser(description='Specify save directors')
parser.add_argument('source')
parser.add_argument('save_dir')
parsing = parser.parse_args()
save_dir = parsing.save_dir
source = parsing.source
print("save_dir:", save_dir)
save_dir = save_dir.rstrip('/')
print("source:", source)


if source == "all":
	source_options = ["intercept", "ottawa-citizen", "cosmo", "national-post"]
else:
	assert source in ["intercept", "ottawa-citizen", "cosmo", "national-post"]
	source_options = [source]

for news_source in source_options:
	print("=================================")
	print("SCRAPING: {}".format(news_source))
	timestamp = time.strftime("%F-%H:%M:%S")

	###########################################################################
	# Loading skippable urls

	skippable_urls_fname = "skippable_urls_{}.log".format(news_source)
	if os.path.exists(skippable_urls_fname):
		print("\tLoading existing skippable file set...")
		with open(skippable_urls_fname, "r") as f:
			skippable_urls = []
			for l in f.read().strip().split("\n"):
				l = l.strip()
				if len(l) == 0: continue
				try:
					skippable_urls.append(l.split('\t'))
					assert len(l.split('\t')) == 2
				except:
					print(l)
					exit()
			# possible reasons: "REQ FAIL", "DONE", "NON ENGLISH", "NO PQ", "FEW SENTS", "NO VALID PQ"
			skippable_urls = set([url for reason, url in skippable_urls if reason not in ["REQ FAIL"]])
	else:
		print("\tNo existing skippable file set")
		skippable_urls = set()


	###########################################################################
	# creating loggers

	skippable_url_log_file = open("skippable_urls_{}.log".format(news_source), "a")
	skip_logger = get_logger(skippable_url_log_file)


	err_log_file = open("errors_{}.log".format(news_source), "a")
	err_logger = get_logger(err_log_file)

	###########################################################################
	# getting url list
	url_list = get_url_list("url_lists/{}_urls.txt".format(news_source), news_source)
	#url_list = url_list[:5]

	###########################################################################
	# scraping!

	full_save_dir = f"{save_dir}/{news_source}/"
	if not os.path.isdir(full_save_dir):
		print("\tCreating directory:", full_save_dir)
		os.mkdir(full_save_dir)

	if news_source == "intercept":
		intercept_scraper.scrape(url_list, save_dir=full_save_dir, err_logger=err_logger, skip_logger=skip_logger, skippable_urls=skippable_urls)

	elif news_source == "ottawa-citizen":
		ottawa_citizen_scraper.scrape(url_list, save_dir=full_save_dir, err_logger=err_logger, skip_logger=skip_logger, skippable_urls=skippable_urls)
	
	elif news_source == "cosmo":
		cosmo_scraper.scrape(url_list, save_dir=full_save_dir, err_logger=err_logger, skip_logger=skip_logger, skippable_urls=skippable_urls)
	
	elif news_source == "national-post":
		national_post_scraper.scrape(url_list, save_dir=full_save_dir, err_logger=err_logger, skip_logger=skip_logger, skippable_urls=skippable_urls)

	err_log_file.close()
	skippable_url_log_file.close()