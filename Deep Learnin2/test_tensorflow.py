import pandas as pd
import numpy as np
import os

import tensorflow as tf

#######Prepare data set ##########
csv_dir = 'D:\python\My Deep learning' 	# your root dataset			
df = pd.read_csv(os.path.join(csv_dir, 'example_2_layer.csv'), dtype=np.float32) 

data_X = df[['X1', 'X2', 'X3']].values 	# training dataset

# correct answers will go here
data_Y = df['Y'].values.reshape(-1,1) 	# convert to vector			

X = tf.constant(data_X)
Y_ = tf.constant(data_Y)
##################################

#######Prepare neural network on Tensorflow graph
tf.set_random_seed(0)

totalFeatures = data_X.shape[1]  # column number
L = 5		# Number nodes of layer1
M = 1		# Number nodes of layer2

W1 = tf.Variable(tf.truncated_normal([totalFeatures, L], stddev=0.1)) 
B1 = tf.Variable(tf.zeros([L]))
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.zeros([M]))

Y1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)
Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)

loss = tf.reduce_mean(tf.square(Y2 - Y_))

# training step, learning rate = 0.1
learning_rate = 0.1
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
##################################

##### Trainin here ##########
w1, w2, b1, b2 = (0, 0, 0, 0)
for step in range(15000):
	_,w1, w2, b1, b2, l, predictions = sess.run([train, W1, W2, B1, B2, loss, Y2])
	
	if (step % 2000 == 0):
		print('Loss at step %d: %f' % (step, l))
		predictions = 1*(predictions > 0.5)
		accuracy = 100.0 * np.sum(predictions == data_Y)/ len(predictions)
		print('Training accuracy: %.1f%%' % accuracy)
		
##############################
### for test #####
if __name__ == '__main__':
	print('All input/output dataset')
	print(df)
	
	X1 = df['X1'] 
	X2 = df['X2'] 
	X3 = df['X3']
	
	#X1 or X2
	temp = np.logical_or(X1, X2)	
	#(X1 or X2) xor X3
	result = np.logical_xor(temp, X3)
	print('Result of X1 or X2 xor X3')
	print(result)
	
	#compare with df
	#1*np.logical_xor(temp, X3) == df['Y'])
	
	def sigmoid(x):
		return 1/(1+np.exp(-x))
	
	new_X = [1,1,0]		# input for testing
	Ouput_layer1 = sigmoid(np.matmul(new_X,w1) + b1)
	Ouput_layer2 = sigmoid(np.matmul(Ouput_layer1,w2) + b2)

	print('\nFor testing')
	print('\nInput:', new_X)
	print('\nWeight for layer 1:\n', w1)
	print('\nWeight for layer 2:\n', w2)
	print('\nOutput values of layer 1:\n', Ouput_layer1)
	print('\nOutput values of layer 2:\n', Ouput_layer2)

	answer = 1*(Ouput_layer2 > 0.5)
	print('\nAnswer is : %d' % answer[0] )
