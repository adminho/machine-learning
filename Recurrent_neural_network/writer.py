# code reference from Keras example : https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop

import os.path
import shutil	
import numpy as np

# copy this function from: https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
def _get_probIndex(preds, temperature=1.0):
	# helper function to sample an index from a probability array
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)

# For example 1
def build_model1(max_seqlen, encoding_len):
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

# For example 2	
def build_model2(max_seqlen, encoding_len):
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

# For example 3
def build_model3(max_seqlen, encoding_len):
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

TEMP_PATH = 'temp'
if os.path.exists(TEMP_PATH):
	shutil.rmtree(TEMP_PATH)	
os.makedirs(TEMP_PATH)
	
def get_trainer(content, tokens, max_seqlen, build_model, step=1):
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
	# batch_seqtokens => encoded to X
	# next_tokens => encoded to y
	# encoding_len => the length of a encoded token vector
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
	
	# write text to files
	def __write_text__(file_name, generate_text):		
		file = open(os.path.join(TEMP_PATH, file_name) ,"w") 			
		text = ''.join(generate_text)
		print('\n**** Generate text *****\n', text)
		file.write(text)	
		file.close()	
				
	# Train		
	def trainer(write_tofilename, num_epochs=10, diversity_list = [0.2, 0.5, 1.0, 1.2]):		
		model.fit(X, y,
			  batch_size=25,
			  epochs=num_epochs, verbose=0) # verbose = 1, 2 print a progress status 
		
		generate_list =[[]] * len(diversity_list)
		# start_index = random.randint(0, len(content) - max_seqlen - 1)		
		for index, diversity in enumerate(diversity_list):	# many diversity			
			print('Tesing with diversity:', diversity)		
			# for begining input			  
			seq_tokens =content[0:max_seqlen]		
			#seq_tokens =content[start_index: start_index + max_seqlen]		
			generate_text = seq_tokens
			print("Generate with begining tokens: ", seq_tokens)
			
			for i in range(0, len(content) - max_seqlen):	
				# One-hot encoding
				seq_encode = np.zeros((1, max_seqlen, encoding_len))		
				for t, token in enumerate(seq_tokens):				
					seq_encode[0, t, token_indices[token]] = 1	
				preds = model.predict(seq_encode, verbose=0)[0] # Output shape is [1, encoding_len]
				next_index = _get_probIndex(preds, diversity)
				next_char = indices_token[next_index]			
				generate_text = np.append(generate_text, next_char)
				seq_tokens = np.append(seq_tokens[1:], next_char)				
			
			file_name =  "diversity_"+ str(diversity) + "_"  + write_tofilename
			# override old file
			__write_text__(file_name, generate_text) 	# write a file for each diversity			
			generate_list[index].append(generate_text) 	# for visual only
		
		return generate_list
	#################### ending trainer function #################
	
	return	trainer