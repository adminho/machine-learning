import pandas as pd
import numpy as np
import os
import tensorflow as tf

#######Prepare data set ##########
csv_dir = 'D:/MyProject/machine-learning/Neural network' 	# your root path of a dataset file					
df = pd.read_csv(os.path.join(csv_dir, 'example_2_layer.csv'), dtype=np.float32) 
data_X = df[['X1', 'X2', 'X3']].values 	# training input

# correct answers will go here
data_Y = df['Y'].values.reshape(-1,1) 	# convert to the vector			
##################################

#######Create neural network architecture with Tensorflow graph
tf.set_random_seed(0)

totalFeatures = data_X.shape[1]  # column number (In example is  3)
L = 5		# Number nodes (neurons) of layer1
M = 1		# Number nodes (neurons) of layer2

X = data_X	# 8 x 3 (number_samples x 3)
Y_ = data_Y	# 8 x 1 (number_samples x 3)

# Declare variables (weight and bias) that updated when training your model
W1 = tf.Variable(tf.truncated_normal([totalFeatures, L], stddev=0.1)) 
B1 = tf.Variable(tf.zeros([L]))
W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.zeros([M]))

# 2 fully connected layer
L1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)  # output: 8 x 5 (number_samples x L)
Predict = tf.nn.sigmoid(tf.matmul(L1, W2) + B2) # prediction output, 8 x 1 (number_samples x M)

# for training model
loss = tf.reduce_mean(tf.square(Predict - Y_))  # loss = Predict - correct answers
# training step, learning rate = 0.1
learning_rate = 0.1
# use gradient descent algorithm to update weight and bias (W1, W2, B1, B2)
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
######Finish create the graph in Tensorflow (Not just run)##################

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # run all variables
##################################

##### Training here ##########
w1, w2, b1, b2 = (0, 0, 0, 0)
for step in range(30000): # backpropagation training
	# run neural network model  
	_,w1, w2, b1, b2, pred, l = sess.run([train, W1, W2, B1, B2, Predict, loss])
	
	# If pred more than 0.5, answer is 1
	# If pred less than 0.5, answer is 0
	# pred vector: size is 8 x 1 (number_samples x 1)
    # In python, True is 1 and False is 0
	pred = 1*(pred > 0.5)

	# accuracy of the trained model
	accuracy = np.mean(1 - np.abs(pred - Y_)) * 100
	if accuracy == 100: # finish
		print('\nTraining accuracy: %.1f%%' % accuracy)
		print('Finish learning at step: %d' % step)
		break
		
	if (step % 3000 == 0): # for debug
		print('\nTraining accuracy: %.1f%%' % accuracy)
		print('Loss at step %d: %f' % (step, l))		
		
##############################
### for test #####
if __name__ == '__main__':
	print('\nAll input/output dataset')
	print(df)
	
	X1 = df['X1'] 
	X2 = df['X2'] 
	X3 = df['X3']
	
	# X1 or X2
	temp = np.logical_or(X1, X2)	
	# (X1 or X2) xor X3
	result = np.logical_xor(temp, X3)
	print('\nLogic result: X1 or X2 xor X3')
	print(result)
	
	#compare with df
	#1*np.logical_xor(temp, X3) == df['Y'])
	
	def sigmoid(x):
		return 1/(1+np.exp(-x))
	
	new_X = [0,1,1]	 # input for testing

	# I will calculate the logic: (X1 or X2) xor X3
	Ouput_layer1 = sigmoid(np.matmul(new_X,w1) + b1)
	Ouput_layer2 = sigmoid(np.matmul(Ouput_layer1,w2) + b2)

	print('\nFor testing')
	print('\nInput:', new_X)
	print('\nWeight for layer 1:\n', w1)
	print('\nWeight for layer 2:\n', w2)
	print('\nOutput values of layer 1:\n', Ouput_layer1)
	print('\nOutput values of layer 2:\n', Ouput_layer2)

	answer = 1*(Ouput_layer2 > 0.5)
	print('\nAnswer is ', answer[0] )