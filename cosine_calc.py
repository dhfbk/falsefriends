import re
import os
from fasttext import FastVector
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
import operator
import sys
import codecs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("lang", help="Language 1 (modified)")
parser.add_argument("skip_lang", help="Language to skip")
parser.add_argument("lang_p_vec", help="Input vectors for pivot language")
parser.add_argument("lang_vec", help="Input vectors for lang")
parser.add_argument("synonyms", help="Synonyms TSV file")
parser.add_argument("test", help="Test file")
parser.add_argument("output", help="Output folder")
args = parser.parse_args()

synonyms_file = args.synonyms
lang = args.lang
skip_lang = args.skip_lang
out_folder = args.output
pairs_file = args.test

### Create synonyms dictionary
synonyms_dict=dict()
syn_file  = open(synonyms_file, "r")
lines = syn_file.readlines()
for line in lines:
	line = re.sub(r'\n', '', line)
	w1, w2 = line.split('\t')
	synonyms_dict[w1] = w2.split(',')

fr_dictionary = FastVector(vector_file=args.lang_vec)
it_dictionary = FastVector(vector_file=args.lang_p_vec)

# Start

out_file_name = out_folder + "/skip" + skip_lang + "_" + lang + ".txt"

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

out_file = codecs.open(out_file_name, 'w', "utf-8")

vec_file_name = re.sub(r'\.txt', '', out_file_name)
vec_file_name = vec_file_name + "_vec.txt"
vec_file = codecs.open(vec_file_name, 'w', "utf-8")

vec_file_nosyn_name = re.sub(r'\.txt', '', out_file_name)
vec_file_nosyn_name = vec_file_nosyn_name + "_nosyn_vec.txt"
vec_file_nosyn = codecs.open(vec_file_nosyn_name, 'w', "utf-8")

vec_file_feat_name = re.sub(r'\.txt', '', out_file_name)
vec_file_feat_name = vec_file_feat_name + "_feats_vec.txt"
vec_file_feat = codecs.open(vec_file_feat_name, 'w', "utf-8")

vec_file_feat7_name = re.sub(r'\.txt', '', out_file_name)
vec_file_feat7_name = vec_file_feat7_name + "_feats7_vec.txt"
vec_file_feat7 = codecs.open(vec_file_feat7_name, 'w', "utf-8")

vec_file_both_name = re.sub(r'\.txt', '', out_file_name)
vec_file_both_name = vec_file_both_name + "_both_vec.txt"
vec_file_both = codecs.open(vec_file_both_name, 'w', "utf-8")

out_file.write("class\tit_word\t"+lang+"_word")
out_file.write("\tcosine_inflected")
out_file.write("\tmean_syn_cosine\tmax_syn_cosine")
out_file.write("\tbest_synonym\tbest_syn_cosine_with_IT\tcosine_best_synonym_with_FR\tsynonims\n")

with open(pairs_file) as f:
	lines = f.readlines()
	for line in lines:
		try:
			line = re.sub(r'\n', '', line)
			annotation, it_w, fr_w = line.split("\t")
			if it_w not in it_dictionary:
				it_w = re.sub(r'-', '', it_w.lower())
			if fr_w not in fr_dictionary:
				fr_w = re.sub(r'-', '', fr_w.lower())
			
		except ValueError:
			continue

		if it_w not in it_dictionary:
			print(it_w+" - "+fr_w+" - it word not found")
			if (lang == skip_lang):
				continue

		if fr_w not in fr_dictionary:
			print(it_w+" - "+fr_w+" - "+lang+" word not found")
			if (lang == skip_lang):
				continue

		# Cosine between words
		try:
			cosine = FastVector.cosine_similarity(it_dictionary[it_w], fr_dictionary[fr_w])
		except KeyError:
			cosine = 1
		
		# Synonyms list
		synonyms_list=[]
		if it_w in synonyms_dict:
			synonyms_list=synonyms_dict[it_w]

		if len(synonyms_list)<1:
			synonyms_list.append(it_w)

		# Array containing all cosines from synonyms
		synonyms_cosine_list=[]
		for s in synonyms_list:
			if s not in it_dictionary:
				continue
			try:
				syn_cosine = FastVector.cosine_similarity(it_dictionary[s], fr_dictionary[fr_w])
			except KeyError:
				syn_cosine = 1
			synonyms_cosine_list.append(syn_cosine)

		# Synonyms cosine max and avg
		if len(synonyms_cosine_list) < 1:
			synonyms_cosine_list.append(cosine)
		synonyms_average_cosine=np.mean(synonyms_cosine_list)
		synonyms_max_cosine=np.max(synonyms_cosine_list)

		# Nearest synonym (same language)
		synonyms_similarity = dict()
		for s in synonyms_list:
			if s not in it_dictionary:
				continue
			try:
				syn_monolingual = FastVector.cosine_similarity(it_dictionary[s], it_dictionary[it_w])
			except KeyError:
				syn_monolingual = 1
			synonyms_similarity[s] = syn_monolingual
		if len(synonyms_similarity) < 1:
			synonyms_similarity[it_w] = 1
		best_synonym=max(synonyms_similarity.items(), key=operator.itemgetter(1))[0]

		# Nearest cosine (other language)
		try:
			best_syn_cosine = FastVector.cosine_similarity(it_dictionary[best_synonym], fr_dictionary[fr_w])
		except KeyError:
			best_syn_cosine = 1


		out_file.write(annotation+"\t"+it_w+"\t"+fr_w)
		out_file.write("\t"+str(cosine))
		out_file.write("\t"+str(synonyms_average_cosine)+"\t"+str(synonyms_max_cosine))
		out_file.write("\t"+best_synonym+"\t"+str(synonyms_similarity[best_synonym])+"\t"+str(best_syn_cosine)+"\t"+str(synonyms_list)+"\n")
		out_file.flush()

		# Features
		if "co" in annotation:
			vec_file_feat.write("0")
		if "ff" in annotation:
			vec_file_feat.write("1")
		vec_file_feat.write(" 1:"+str(cosine))
		vec_file_feat.write(" 5:"+str(synonyms_average_cosine))
		vec_file_feat.write(" 6:"+str(synonyms_max_cosine))
		vec_file_feat.write(" 7:"+str(synonyms_similarity[best_synonym]))
		vec_file_feat.write(" 8:"+str(best_syn_cosine))
		vec_file_feat.write("\n")

		# Features (without 7)
		if "co" in annotation:
			vec_file_feat7.write("0")
		if "ff" in annotation:
			vec_file_feat7.write("1")
		vec_file_feat7.write(" 1:"+str(cosine))
		vec_file_feat7.write(" 5:"+str(synonyms_average_cosine))
		vec_file_feat7.write(" 6:"+str(synonyms_max_cosine))
		vec_file_feat7.write(" 8:"+str(best_syn_cosine))
		vec_file_feat7.write("\n")

		# Features (no synonyms)
		if "co" in annotation:
			vec_file_nosyn.write("0")
		if "ff" in annotation:
			vec_file_nosyn.write("1")
		vec_file_nosyn.write(" 1:"+str(cosine))
		vec_file_nosyn.write("\n")


		# Concatenation
		if (fr_w in fr_dictionary) and (it_w in it_dictionary):
			feature_counter=1
			if "co" in annotation:
				vec_file.write("0")
			if "ff" in annotation:
				vec_file.write("1")
			for v in it_dictionary[it_w]:
				vec_file.write(" "+str(feature_counter)+":"+str(v))
				feature_counter=feature_counter+1
			for v in fr_dictionary[fr_w]:
				vec_file.write(" "+str(feature_counter)+":"+str(v))
				feature_counter=feature_counter+1

			vec_file.write("\n")

		# Output both
		if (fr_w in fr_dictionary) and (it_w in it_dictionary):
			feature_counter=1
			if "co" in annotation:
				vec_file_both.write("0")
			if "ff" in annotation:
				vec_file_both.write("1")
			for v in it_dictionary[it_w]:
				vec_file_both.write(" "+str(feature_counter)+":"+str(v))
				feature_counter=feature_counter+1
			for v in fr_dictionary[fr_w]:
				vec_file_both.write(" "+str(feature_counter)+":"+str(v))
				feature_counter=feature_counter+1
			vec_file_both.write(" 1201:"+str(cosine))
			vec_file_both.write(" 1205:"+str(synonyms_average_cosine))
			vec_file_both.write(" 1206:"+str(synonyms_max_cosine))
			vec_file_both.write(" 1207:"+str(synonyms_similarity[best_synonym]))
			vec_file_both.write(" 1208:"+str(best_syn_cosine))
			vec_file_both.write("\n")
