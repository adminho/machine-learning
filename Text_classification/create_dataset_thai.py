# Visit nectec website: https://www.nectec.or.th/corpus/index.php?league=pm
# and download list files including
# 1) article.zip
# 2) encyclopedia.zip
# 3) news.zip
# 4) novel.zip
#
# for testing
# 5) TEST_100K.txt

# I use TNC2_freq-5000.xls  (download at: http://www.arts.chula.ac.th/~ling/TNC/category.php?id=58&)
# reference 5000 thai words is found frequency 
# but I tranfered this file to "TNC2_freq-5000.csv"

import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import pickle
import numpy as np
import re

import deepcut

PICKLE_FOLDER = "pickle"
if os.path.exists(PICKLE_FOLDER) == False:
	os.makedirs(PICKLE_FOLDER)
	
# List of the most words found frequency in Thai language
FILE_FREQ_WORD="TNC2_freq-5000.csv"
df_word2index = pd.read_csv(FILE_FREQ_WORD, index_col='word', usecols=['word', 'index'])
# save this dataframe to the pickle file
pickle.dump( df_word2index, open( join(PICKLE_FOLDER, "word2index.p"), "wb" ) )

# dictionary of words that found frequency
dict_word = df_word2index.to_dict(orient='dict')
dict_word = dict_word['index']
word_freqList = list(dict_word.keys())

# this function is very slow (plan to improve it)
def get_index(word_list):
	word_seqIndex = []
	# convert words into the index
	for word in word_list:				
		# select word that found frequency and then get index (from TNC2_freq-5000.csv)
		index = '' 
		if word in word_freqList:			
			#index = word_freqList.index(word)+1
			index = dict_word[word]
		# otherwise					
		elif re.search('<NE>.*</NE>', word) is not None: 		# for Named Entity 			
			index = 5001
		elif re.search('<AB>.*</AB>', word) is not None: 		# for Abbreviation			
			index = 5002
		elif re.search('<POEM>.*</POEM>', word) is not None: 	# for Poem			
			index = 5003			
		else:# rarely found			
			index = 0			
		word_seqIndex.append(index)
	return word_seqIndex

def dataset2index(source_dir, pickle_file=None):
	# get all file names
	all_fileNames = [join(source_dir, file) for file in listdir(source_dir)]
	contentList = []
	
	for file in all_fileNames:
		with open( file, "r", encoding="utf8") as f:
			word_list = f.read().split('|')	
			word_seqIndex = get_index(word_list) # convert words into the index
		contentList.append(word_seqIndex)
	
	print("Total content: ", np.shape(contentList) ) # shape output is (number files, )
	assert np.shape(contentList)[0]  == len(all_fileNames)		
	
	if pickle_file is not None:
		# save to a pickle file
		pickle.dump( contentList, open( join(PICKLE_FOLDER, pickle_file), "wb" ) )
		# for the example how to load a pickle file
		# contentList = pickle.load( open( "article_thai.p", "rb" ) )
	return contentList

def content2index(file_name):
	f = open( file_name, "r", encoding="utf8")
	sentences = f.read().split(" ") # split "space" char	
	conntentIndex = []	
	for s in sentences:	
		# tokenize thai word
		word_list = deepcut.tokenize(s)
		# and then convert words to indexs
		word_seqIndex = get_index(word_list)		
		conntentIndex = conntentIndex + word_seqIndex				
	return conntentIndex

def create_classify_dataset(dataset_list):
	X_trainList, X_testList, Y_trainList, Y_testList = [], [], [], []
	for index, X in enumerate(dataset_list):
		total_example = len(X)
		# label 0: 	"article", label 1: "encyclopedia", label 2: "news", label 4: "novel"
		Y_label = [index] * total_example # create label followed by index (class name)
		
		split_index = int(total_example * 0.80)
		X_train, X_test = X[0:split_index], X[split_index:]
		Y_train, Y_test = Y_label[0:split_index:], Y_label[split_index:]
		
		X_trainList = X_trainList + X_train
		X_testList = X_testList + X_test
		Y_trainList = Y_trainList + Y_train
		Y_testList = Y_testList + Y_test
	
	return X_trainList, X_testList, Y_trainList, Y_testList 

def load_dataset():
	dict = pickle.load( open( join(PICKLE_FOLDER,"classify_text.p"), "rb" ) )
	X_train, X_test, Y_train, Y_test = dict['X_trainList'], dict['X_testList'], dict['Y_trainList'], dict['Y_testList']
	return X_train, X_test, Y_train, Y_test 

def load_dataset_unknown(pickle_file):
	return pickle.load( open( join(PICKLE_FOLDER, pickle_file), "rb" ) )
	
def main():	
	output_path = ''
	SOURCE_PATH="D:/MyProject/Big-datasets/dataset-thai-word/" # You can change here this path for your datasets
	split_char = ' ' # file dataset use '|'	to seperate words in the content
	
	print("\n------- Convert 'article' dataset to set of indexs ----------")
	article = dataset2index( join(SOURCE_PATH, "article") , join(output_path,"article_thai.p"))
	
	print("\n------- Convert 'encyclopedia' dataset to set of indexs ----------")
	encyclopedia = dataset2index( join(SOURCE_PATH, "encyclopedia") ,join(output_path,"encyclopedia_thai.p"))
	
	print("\n------- Convert 'news' dataset to set of indexs----------")
	news = dataset2index( join(SOURCE_PATH, "news") , join(output_path,"news_thai.p"))
	
	print("\n------- Convert 'novel' dataset to set of indexs----------")
	novel =dataset2index( join(SOURCE_PATH, "novel") , join(output_path,"novel_thai.p"))

	print("\n------------create dataset for classify task----------")
	dataset_list = [article, encyclopedia, news, novel]
	X_trainList, X_testList, Y_trainList, Y_testList  = create_classify_dataset(dataset_list)
	assert len(X_trainList) + len(X_testList)  == len(Y_trainList) +  len(Y_testList) 	
	assert len(X_trainList) + len(X_testList)  == len(article) +  len(encyclopedia) + len(news) +  len(novel)	
		
	dict = {}
	dict['X_trainList'] = X_trainList
	dict['X_testList'] = X_testList
	dict['Y_trainList'] = Y_trainList
	dict['Y_testList'] = Y_testList
	# save dictionary to a pickle file
	pickle.dump( dict, open( join(PICKLE_FOLDER, "classify_text.p"), "wb" ) )
	
	print("\n------------ Convert unknown dataset (for test) to set of indexs ----------")
	file_name = "TEST_NOVEL.txt"
	contentList = content2index(file_name)	
	# save to a pickle file
	pickle.dump( contentList, open( join(PICKLE_FOLDER, file_name + ".p"), "wb" ) )	
	
if __name__ == "__main__":
	main()