# My idea from this example : https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop

import os.path
import shutil	
import numpy as np

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

class Vocabulary():
	def __init__(self, tokens):
		# Vocabulary
		# for converting all tokens to index
		self.token2indices = dict((token, i) for i, token in enumerate(tokens))
		# for converting all index to a tokens
		self.indices2token = dict((i, token) for i, token in enumerate(tokens))
		self.encoding_len = len(tokens)	

	def encode_target(self, next_tokens):
		# One-hot encoding		
		Y_encoded = np.zeros((len(next_tokens), self.encoding_len))	
		for i, tokens in enumerate(next_tokens):		
			Y_encoded[i, self.token2indices[tokens]] = 1
		return Y_encoded
	
	def encode_input(self, batch_tokens):
		# One-hot encoding
		total, seq_len = np.shape(batch_tokens)		
		X_encoded = np.zeros((total, seq_len, self.encoding_len))
		for i, tokens in enumerate(batch_tokens):
			for t, token in enumerate(tokens):
				X_encoded[i, t, self.token2indices[token]] = 1
		return X_encoded
	
	# copy this function from: https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
	def decode_predict(self, preds, temperature=1.0):
		# helper function to sample an index from a probability array
		preds = np.asarray(preds).astype('float64')
		preds = np.log(preds) / temperature
		exp_preds = np.exp(preds)
		preds = exp_preds / np.sum(exp_preds)
		probas = np.random.multinomial(1, preds, 1)
		return np.argmax(probas)
	
def get_trainer(content, tokens, max_seqlen, build_model, step=1):
	# clear up tokens duplicated
	tokens = sorted(list(set(content)))
	encoding_len = len(tokens)	
	print("\ntokens:\n", tokens)
	print("\ntotal tokens: ", encoding_len)
	print("sequences length: ", max_seqlen)
		
	vocab = Vocabulary(tokens)
	
	print('Preparing the input and target...')
	# Preparing the input and target
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
	X = vocab.encode_input(batch_seqtokens)
	y = vocab.encode_target(next_tokens)	
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
				seq_encoded = vocab.encode_input([seq_tokens]) # input shape is [1, number of tokens, encoding_len ]				
				preds = model.predict(seq_encoded, verbose=0)[0] # Output shape is [1, encoding_len]
				
				next_index = vocab.decode_predict(preds, diversity)
				next_char = vocab.indices2token[next_index]			
				generate_text = np.append(generate_text, next_char)
				seq_tokens = np.append(seq_tokens[1:], next_char)				
			
			file_name =  "diversity_"+ str(diversity) + "_"  + write_tofilename
			# override old file
			__write_text__(file_name, generate_text) 	# write a file for each diversity			
			generate_list[index].append(generate_text) 	# for visual only
		
		return generate_list
	#################### ending trainer function #################
	
	return	trainer