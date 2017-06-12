# I modified from : https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random

# Ignore case-sensitive
source_code = open('index.html', encoding="utf8").read().lower()

# I'm split to an array easily (In practise, don't it)
source_code = source_code.split() # default is space to split 
print('source code exampe:')
print(source_code)
print('\ncorpus length:', len(source_code))

# to reduce duplicate words
tokens = sorted(list(set(source_code)))
ENCODING_LEN = len(tokens)
# looking sequences of words (not characters)		
print('total tokens:', ENCODING_LEN)

# for converting a token to index
token_indices = dict((c, i) for i, c in enumerate(tokens))
# for converting index to a token
indices_token = dict((i, c) for i, c in enumerate(tokens))

# cut the source_code in semi-redundant sequences of MAX_SEQ_LEN characters
MAX_SEQ_LEN = 2
batch_seq_tokens = []
next_tokens = []
for i in range(0, len(source_code) - MAX_SEQ_LEN):
    batch_seq_tokens.append(source_code[i: i + MAX_SEQ_LEN])
    next_tokens.append(source_code[i + MAX_SEQ_LEN])
	
print('len batch_seq_tokens:', len(batch_seq_tokens))
print('len next_tokens:', len(next_tokens))

print('Vectorization...')
# One-hot encoding 
# batch_seq_tokens is encoded to X
# next_tokens is encoded to y
# ENCODING_LEN is: the length of a encoded token vector
X = np.zeros((len(batch_seq_tokens), MAX_SEQ_LEN, ENCODING_LEN))
y = np.zeros((len(batch_seq_tokens), ENCODING_LEN))
for i, seq_tokens in enumerate(batch_seq_tokens):
    for t, token in enumerate(seq_tokens):
        X[i, t, token_indices[token]] = 1
    y[i, token_indices[next_tokens[i]]] = 1

assert X.shape == (24, 2, 22)
assert y.shape == (24, 22)

# build the model: a single LSTM
print('Build model...')
model = Sequential()
# Input size: (None, 2, 22) = (bath_num, sequences_num, dim_input)
model.add(LSTM(20, input_shape=(MAX_SEQ_LEN, ENCODING_LEN), return_sequences=True))
model.add(LSTM(20, return_sequences=False))
# Fully-connected layer
model.add(Dense(ENCODING_LEN))
model.add(Activation('softmax'))

print(model.summary())
"""_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm_1 (LSTM)                (None, 2, 20)             3440
_________________________________________________________________
lstm_2 (LSTM)                (None, 20)                3280
_________________________________________________________________
dense_1 (Dense)              (None, 22)                462
_________________________________________________________________
activation_1 (Activation)    (None, 22)                0
=================================================================
Total params: 7,182
Trainable params: 7,182
Non-trainable params: 0
_________________________________________________________________
"""

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# copy this function from: https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
def get_probIndex(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# One-hot encoding			  
def encode_X(seq_tokens):
	sequences_encode = np.zeros((1, MAX_SEQ_LEN, ENCODING_LEN))
	for t, token in enumerate(seq_tokens):    
		sequences_encode[0, t, token_indices[token]] = 1
	return sequences_encode

# Train	
for iteration in range(1, 15):    
    print('Iteration %s\n' % iteration)
    model.fit(X, y,
              batch_size=25,
              epochs=10, verbose=0) # verbose = 1, 2 print a progress status 

# for begining			  
generate ="<html> <head>"
seq_tokens = generate.split()
print("Tokens for begining:")
print(seq_tokens) # ['<html>', '<head>']

for i in range(0, len(source_code) - MAX_SEQ_LEN):    
	sequences_encode = encode_X(seq_tokens)	
	preds = model.predict(sequences_encode, verbose=0)[0]
	next_index = get_probIndex(preds)
	next_char = indices_token[next_index]
	generate += ' ' + next_char
	seq_tokens = np.append(seq_tokens[1:],next_char)         

print('Generate text:\n', generate)