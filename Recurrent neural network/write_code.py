# I modified from : https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random

source_code = open('index.php', encoding="utf8").read().lower()
# I'm split to array easily (In practise, don't it)
source_code = source_code.split() # default is space to split 
print('source code exampe:')
print(source_code)
print('corpus length:', len(source_code))

# to reduce duplicate words
tokens = sorted(list(set(source_code)))
num_tokens = len(tokens)		
print('total tokens:', num_tokens)

# for converting a token to index
char_indices = dict((c, i) for i, c in enumerate(tokens))
# for converting index to a token
indices_char = dict((i, c) for i, c in enumerate(tokens))

# cut the source_code in semi-redundant sequences of maxlen characters
maxlen = 2
batch_seq_tokens = []
next_tokens = []
for i in range(0, num_tokens - maxlen):
    batch_seq_tokens.append(source_code[i: i + maxlen])
    next_tokens.append(source_code[i + maxlen])
	
print('len batch_seq_tokens:', len(batch_seq_tokens))
print('len next_tokens:', len(next_tokens))

print('Vectorization...')
# One-hot encoding 
# batch_seq_tokens is encoded to X
# next_tokens is encoded to y
# num_tokens is: the length of a encoded token vector
X = np.zeros((len(batch_seq_tokens), maxlen, num_tokens), dtype=np.bool)
y = np.zeros((len(batch_seq_tokens), num_tokens), dtype=np.bool)
for i, seq_tokens in enumerate(batch_seq_tokens):
    for t, token in enumerate(seq_tokens):
        X[i, t, char_indices[token]] = 1
    y[i, char_indices[next_tokens[i]]] = 1

print("X shape", X.shape)
print("y.shape", y.shape)

# build the model: a single LSTM
print('Build model...')
model = Sequential()
# Input size: (None, 2, 27) = (bath_num, sequences_num, dim_input)
# Ouput size: (None, 2, 20)
model.add(LSTM(20, input_shape=(maxlen, num_tokens), return_sequences=True))
# Ouput size: (None, 20)
model.add(LSTM(20, return_sequences=False))

# Fully-connected layer
# Ouput size: (None, 27)
model.add(Dense(num_tokens))
model.add(Activation('softmax'))
print(model.summary())

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def get_index(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Train	
for iteration in range(1, 13):    
    print('Iteration', iteration)
    model.fit(X, y,
              batch_size=25,
              epochs=10,verbose=0) # verbose 1 print a progress status

# One-hot encoding			  
sequences_token = ['</head>', '<body>']
sequences_encode = np.zeros((1, maxlen, num_tokens))
for t, char in enumerate(sequences_token):    
    sequences_encode[0, t, char_indices[char]] = 1
				
preds = model.predict(sequences_encode, verbose=0)[0]
next_index = get_index(preds)
next_char = indices_char[next_index]

print('Tokens list:', sequences_token)
print('Answer:',  next_char)