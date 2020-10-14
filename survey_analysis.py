
import numpy as np


with open("survey_analysis.tsv", "r") as f:
	lines = f.read().split("\n")


lines = [l.split("\t") for l in lines]
column_names = lines[0]






# first, lets calculate the average for each model
value_col_indices = [i for i in range(len(column_names)) if '/' in column_names[i]]

model_col_index = column_names.index("Model")
models = list(set([line[model_col_index] for line in lines[1:]]))

per_model = dict(zip(models, [[] for _ in models]))

for line in lines[1:]:
	model = line[model_col_index]
	values = [line[i] for i in value_col_indices if line[i] != '']
	values = [int(v) for v in values]
	per_model[model].extend(values)


print("Model\tTotal Average")
for model in models:
	print("{}\t{:.2f}".format(model, np.average(per_model[model])))


# for each article... sort the models by performance
article_col_index = column_names.index("Article")
article_names = list(set([line[article_col_index] for line in lines[1:]]))

# article -> (model, score)
per_article_results = dict(zip(article_names, [[] for _ in article_names]))

for line in lines[1:]:
	article = line[article_col_index]
	model = line[model_col_index]
	values = [line[i] for i in value_col_indices if line[i] != '']
	values = [int(v) for v in values]
	avg = np.average(values)
	per_article_results[article].append((model, avg))

model_rankings = dict(zip(models, [[] for _ in models]))
for article in article_names:
	results = per_article_results[article]
	# sort model by performance on article (from best to worst)
	results = sorted(results, key=lambda el:-el[1])
	place = 0
	prev_score = float('inf')
	for i in range(len(results)):
		if results[i][1] < prev_score:
			place += 1
		results[i] = (results[i][0], results[i][1], place)
		prev_score = results[i][1]
	for model, avg_score, rank in results:
		model_rankings[model].append(rank)

print()
print("Model\tRank\tPct")
for model in models:
	for place in [1, 2, 3, 4, 5, 6]:
		print("{}\t{}\t{:.1f}%".format(model, place, 100*model_rankings[model].count(place)/len(model_rankings[model])))

print()
print("Model\tAvg. Rank")
for model in models:
	print("{}\t{:.1f}".format(model, np.average(model_rankings[model])))


# want to compute just the following:
#	- average score
#	- average rank
#	- pct time best
print()
print("Model\tAvg. Rating\tAvg. Rank\t1st Place")
res_lines = []
for model in models:
	avg_rating = np.average(per_model[model])
	avg_rank = np.average(model_rankings[model])
	first_place_pct = 100*model_rankings[model].count(1)/len(model_rankings[model])
	res_lines.append((model, avg_rating, avg_rank, first_place_pct))
	#print("{}\t{:.1f}\t{:.1f}\t{:.0f}%".format(model, avg_rating, avg_rank, first_place_pct))

# print results from best to worst model
res_lines = sorted(res_lines, key=lambda el: -el[1])
for model, avg_rating, avg_rank, first_place_pct in res_lines:
	print("{}\t{:.2f}\t{:.2f}\t{:.0f}%".format(model, avg_rating, avg_rank, first_place_pct))