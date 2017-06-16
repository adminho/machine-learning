# I modified from : https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random


# copy this function from: https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
def get_probIndex(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def build_model_example1(max_seq_len, encoding_len):
	model = Sequential()
	# Input size: (bath_num, sequences_num, dim_input)
	model.add(LSTM(20, input_shape=(max_seq_len, encoding_len), return_sequences=True))
	model.add(LSTM(20, return_sequences=False))
	# Fully-connected layer
	model.add(Dense(encoding_len))
	model.add(Activation('softmax'))
	print(model.summary())

	optimizer = RMSprop(lr=0.01)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)
	return model
	
def build_model_example2(max_seq_len, encoding_len):
	#Build LSTM (Long short-term memory)
	model = Sequential()
	# Input size: (bath_num, sequences_num, dim_input)
	model.add(LSTM(100, input_shape=(max_seq_len, encoding_len), return_sequences=True))
	model.add(LSTM(100, return_sequences=False))
	# Fully-connected layer
	model.add(Dense(encoding_len))
	model.add(Activation('softmax'))
	print(model.summary())

	optimizer = RMSprop(lr=0.01)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)
	return model
	
def test_LSTM(content, tokens, max_seq_len, build_model, epochs=10, step=1):
	encoding_len = len(tokens)	
	print("tokens:\n", tokens)
	print("total tokens: ", encoding_len)

	# Vocabulary
	# for converting a token to index
	token_indices = dict((c, i) for i, c in enumerate(tokens))
	# for converting index to a token
	indices_token = dict((i, c) for i, c in enumerate(tokens))

	# cut the content in semi-redundant sequences of max_seq_len characters	
	batch_seq_tokens = []
	next_tokens = []
	for i in range(0, len(content) - max_seq_len, step):
		batch_seq_tokens.append(content[i: i + max_seq_len])
		next_tokens.append(content[i + max_seq_len])
	
	print('len batch_seq_tokens:', len(batch_seq_tokens))
	print('len next_tokens:', len(next_tokens))

	print('Vectorization...')
	# One-hot encoding 
	# batch_seq_tokens is encoded to X
	# next_tokens is encoded to y
	# encoding_len is: the length of a encoded token vector
	X = np.zeros((len(batch_seq_tokens), max_seq_len, encoding_len))
	y = np.zeros((len(batch_seq_tokens), encoding_len))
	for i, seq_tokens in enumerate(batch_seq_tokens):
		for t, token in enumerate(seq_tokens):
			X[i, t, token_indices[token]] = 1
		y[i, token_indices[next_tokens[i]]] = 1

	print("X shape:", X.shape)
	print("Y shape:", y.shape)

	print('Build model...')
	model = build_model(max_seq_len, encoding_len)	

	# Train	
	for iteration in range(0, 15):    
		print('Iteration %s\n' % iteration)
		model.fit(X, y,
              batch_size=25,
              epochs=epochs, verbose=0) # verbose = 1, 2 print a progress status 

	# Testing
	# for begining input			  
	seq_tokens =content[0:max_seq_len]		
	print("Tokens for begining:\n", seq_tokens)
	
	generate = [seq_tokens]
	for i in range(0, len(content) - max_seq_len):    
		# One-hot encoding
		seq_encode = np.zeros((1, max_seq_len, encoding_len))		
		for t, token in enumerate(seq_tokens):    			
			seq_encode[0, t, token_indices[token]] = 1	
		preds = model.predict(seq_encode, verbose=0)[0]
		next_index = get_probIndex(preds)
		next_char = indices_token[next_index]
		generate = np.append(generate, next_char)
		seq_tokens = np.append(seq_tokens[1:], next_char)     
			
	return	generate

if __name__ == "__main__":
	# First example: sequences of words
	# Ignore case-sensitive
	source_code = open('index.html', encoding="utf8").read().lower()	
	print('source code exampe:\n', source_code)	
	print('\ncorpus length:', len(source_code))
	
	# I'm split to an array easily (In practise, don't it)
	source_code = source_code.split() # default is space to split 
	
	# to reduce dupicate
	words = sorted(list(set(source_code)))
	max_seq_len=2
	
	generate = test_LSTM(source_code, words, max_seq_len , build_model_example1)
	text = ''.join(generate)
	print('Generate text:\n', text)
	file = open("testfile.html","w") 
	file.write(text)
	file.close()
	
	# Second example: sequences of characters	
	# Ignore case-sensitive
	# Text from: http://ecomputernotes.com/fundamental/introduction-to-computer/what-is-computer
	content = open('text_eng.txt', encoding="utf8").read().lower()
	print('\nContent example:\n', content)
	print('\ncorpus length:', len(content))

	# to reduce duplicate 			
	chars = sorted(list(set(content)))
	max_seq_len=10
	
	generate = test_LSTM(list(content), chars, max_seq_len , build_model_example2, epochs=20)
	print(len(generate))
	text = ''.join(generate)
	print('Generate text:\n', text)
	file = open("testfile.txt","w") 
	file.write(text)
	file.close()