# code reference from Keras example : https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random

import deepcut

# copy this function from: https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
def get_probIndex(preds, temperature=1.0):
	# helper function to sample an index from a probability array
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)

def build_model_example1(max_seqlen, encoding_len):
	model = Sequential()
	# Input size: (bath_num, sequences_num, dim_input)
	model.add(LSTM(20, return_sequences=True, input_shape=(max_seqlen, encoding_len)))
	model.add(LSTM(20, return_sequences=False))
	# Fully-connected layer
	model.add(Dense(encoding_len))
	model.add(Activation('softmax'))
	print(model.summary())

	optimizer = RMSprop(lr=0.01)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)
	return model
	
def build_model_example2(max_seqlen, encoding_len):
	#Build LSTM (Long short-term memory)
	model = Sequential()
	# Input size: (bath_num, sequences_num, dim_input)
	model.add(LSTM(40, return_sequences=True, input_shape=(max_seqlen, encoding_len)))
	model.add(LSTM(40, return_sequences=False))
	# Fully-connected layer
	model.add(Dense(encoding_len))
	model.add(Activation('softmax'))
	print(model.summary())

	optimizer = RMSprop(lr=0.01)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)
	return model

def build_model_example3(max_seqlen, encoding_len):
	#Build LSTM (Long short-term memory)
	model = Sequential()
	# Input size: (bath_num, sequences_num, dim_input)
	model.add(GRU(40, return_sequences=True, input_shape=(max_seqlen, encoding_len)))
	model.add(GRU(40, return_sequences=False))
	# Fully-connected layer
	model.add(Dense(encoding_len))
	model.add(Activation('softmax'))
	print(model.summary())

	optimizer = RMSprop(lr=0.01)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)
	return model
	
def test(content, tokens, max_seqlen, build_model, epochs=10, step=1, diversity_list = [0.2, 0.5, 1.0, 1.2]):
	# clear up tokens duplicated
	tokens = sorted(list(set(content)))
	encoding_len = len(tokens)	
	print("\ntokens:\n", tokens)
	print("\ntotal tokens: ", encoding_len)
	print("sequences length: ", max_seqlen)
	
	# Vocabulary
	# for converting a token to index
	token_indices = dict((token, i) for i, token in enumerate(tokens))
	# for converting index to a token
	indices_token = dict((i, token) for i, token in enumerate(tokens))

	# cut the content in semi-redundant sequences of max_seqlen characters
	batch_seqtokens = []
	next_tokens = []
	for i in range(0, len(content) - max_seqlen, step):
		batch_seqtokens.append(content[i: i + max_seqlen])
		next_tokens.append(content[i + max_seqlen])
	
	print('len batch_seqtokens:', len(batch_seqtokens))
	print('len next_tokens:', len(next_tokens))

	print('Vectorization...')
	# One-hot encoding 
	# batch_seqtokens is encoded to X
	# next_tokens is encoded to y
	# encoding_len is: the length of a encoded token vector
	X = np.zeros((len(batch_seqtokens), max_seqlen, encoding_len))
	y = np.zeros((len(batch_seqtokens), encoding_len))
	for i, seq_tokens in enumerate(batch_seqtokens):
		for t, token in enumerate(seq_tokens):
			X[i, t, token_indices[token]] = 1
		y[i, token_indices[next_tokens[i]]] = 1

	print("X shape:", X.shape)
	print("Y shape:", y.shape)

	print('Build model...')
	model = build_model(max_seqlen, encoding_len)	

	# Train	
	for iteration in range(0, 15):	
		print('Iteration %s\n' % iteration)
		model.fit(X, y,
			  batch_size=25,
			  epochs=epochs, verbose=0) # verbose = 1, 2 print a progress status 
	
	print("\n-------- Testing -------" )
	generate_list = []
	# start_index = random.randint(0, len(content) - max_seqlen - 1)
	# many diversity
	for diversity in diversity_list:
		print('\n----- diversity-----:', diversity)		
		# for begining input			  
		seq_tokens =content[0:max_seqlen]		
		#seq_tokens =content[start_index: start_index + max_seqlen]		
		generate = seq_tokens
		print("-----Generate with begining tokens: ", seq_tokens)
			
		for i in range(0, len(content) - max_seqlen):	
			# One-hot encoding
			seq_encode = np.zeros((1, max_seqlen, encoding_len))		
			for t, token in enumerate(seq_tokens):				
				seq_encode[0, t, token_indices[token]] = 1	
			preds = model.predict(seq_encode, verbose=0)[0] # Output shape is [1, encoding_len]
			next_index = get_probIndex(preds, diversity)
			next_char = indices_token[next_index]			
			generate = np.append(generate, next_char)
			seq_tokens = np.append(seq_tokens[1:], next_char)	 
			
		generate_list.append(generate) # end each diversity		
	return	generate_list

def tudkum_thai(file_name):
	file = open( file_name, "r", encoding="utf8")
	sentences = file.read().split(" ") # split "space" char	
	content = []	
	for s in sentences:	
		# tokenize thai words
		word_list = deepcut.tokenize(s)		
		content = content + word_list				
	return content

# write text to files
def write_text(postfix_name, generate_list):
	for i, generate in enumerate(generate_list):
		file_name = "%s_%s" % (i+1, postfix_name)
		file = open(file_name,"w") 			
		text = ''.join(generate)
		print('\n**** Generate text *****\n', text)
		file.write(text)
		file.close()

if __name__ == "__main__":
	print("\n-------------- Example 1 --------------------- ")
	# sequences of words
	# Ignore case-sensitive
	source_code = open('index.html', encoding="utf8").read().lower()	
	#print('source code exampe:\n', source_code)	
	print('\ncorpus length:', len(source_code))
		
	# I'm split word in content to a list easily
	source_codeList = source_code.split() # default is space to split 	
	# select words (unique)
	words = sorted(list(set(source_codeList)))	
		
	max_seqlen=2
	generate_list = test(source_codeList, words, max_seqlen, build_model_example1, epochs=10, diversity_list=[1])	
	#write_text("test_code.html", generate_list) 
	
	print("\n-------------- Example 2 --------------------- ")
	# sequences of characters	
	# Ignore case-sensitive
	# Text from: http://ecomputernotes.com/fundamental/introduction-to-computer/what-is-computer
	content = open('text_eng.txt', encoding="utf8").read().lower()
	#print('\nContent example:\n', content)
	print('\ncorpus length:', len(content))

	contentList = list(content)
	# select characters (unique)
	chars = sorted(list(set(content)))	
		
	max_seqlen=10
	generate_list = test(contentList, chars, max_seqlen, build_model_example2, epochs=20, diversity_list=[1])	
	#write_text("test_eng.txt", generate_list) 
	
	print("\n-------------- Example 3 --------------------- ")
	# sequences of thai words
	# tokenize thai words
	contentList = tudkum_thai('TEST_NOVEL_small.txt')
	print('\ncorpus length:', len(contentList))	
	
	# select thai words (unique)
	word_thai = sorted(list(set(contentList)))	
		
	max_seqlen=10
	generate_list = test(contentList, word_thai, max_seqlen, build_model_example3, epochs=20, diversity_list=[1])	
	write_text("test_thai_novel.txt", generate_list) 	