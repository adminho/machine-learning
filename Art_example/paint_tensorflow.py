# Inspiration: http://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html
import numpy as np
import tensorflow as tf
import util as ut

def getTrainModel(): # The model at here	 
	chanel = 3 # Red, Green, Blue in a pixel	
	# Input for training is coordinates (x,y) on a picture
	X = tf.placeholder(tf.float32, [None, 2]) 		# [n sample, x , y ]
	# correct answers will go here (Output for training is RGB value in pixel)
	Y_ = tf.placeholder(tf.float32, [None, chanel])	# [n sample, Red, Green, Blue]
	# All values will feed into X and Y_ later	

	# fully connected layers but last layer has 3 neurons or nodes (3 chanel)
	K = 20  # number of neurons in first fully connected layer
	L = 20  # number of neurons in second fully connected layer
	M = 20  # number of neurons in third fully connected layer
	N = 20  # number of neurons in forth fully connected layer
	O = 20  # number of neurons in fifth fully connected layer
	P = 20  # number of neurons in sixth fully connected layer
	Q = 20	# number of neurons in seventh fully connected layer

	# Weights and bias are initialised 
	W1 = tf.Variable(tf.truncated_normal([2, K], stddev=0.1)) # first layer
	B1 = tf.Variable(tf.ones([K])/chanel)

	W2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
	B2 = tf.Variable(tf.ones([L])/chanel)

	W3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
	B3 = tf.Variable(tf.ones([M])/chanel)

	W4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
	B4 = tf.Variable(tf.ones([N])/chanel)

	W5 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
	B5 = tf.Variable(tf.ones([O])/chanel)

	W6 = tf.Variable(tf.truncated_normal([O, P], stddev=0.1))
	B6 = tf.Variable(tf.ones([P])/chanel)

	W7 = tf.Variable(tf.truncated_normal([P, Q], stddev=0.1)) # seventh layer
	B7 = tf.Variable(tf.ones([Q])/chanel)

	W8 = tf.Variable(tf.truncated_normal([Q, chanel], stddev=0.1)) # last layer
	B8 = tf.Variable(tf.ones([chanel])/chanel)

	# connect all layer together
	Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
	Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
	Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
	Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
	Y5 = tf.nn.relu(tf.matmul(Y4, W5) + B5)
	Y6 = tf.nn.relu(tf.matmul(Y5, W6) + B6)
	Y7 = tf.nn.relu(tf.matmul(Y6, W7) + B7)	
	ColorPredicted = tf.nn.relu(tf.matmul(Y7, W8) + B8) # output at last layer is RGB answer (3 output)

	# accuracy of the trained model
	accuracy = tf.reduce_mean( 1 - tf.abs(ColorPredicted - Y_)) * 100
	# Loss function, use MSE function
	loss = tf.reduce_mean(tf.square(ColorPredicted - Y_))
	# learning rate (values will feed into lr variable later)
	lr = tf.placeholder(tf.float32)
	# set traing alogrithm 
	optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

	# init tensorflow
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)

	def trainModel(step, coor_train, colorData_train):
		# the backpropagation training step			
		max_learning_rate = 0.0100
		min_learning_rate = 0.0001
		# learning rate decay
		decay_speed = 20000.0
		learning_rate = min_learning_rate + \
						(max_learning_rate - min_learning_rate) * np.exp(-step/decay_speed)
				
		# all values will put into all placeholder varibles (X and Y_)
		# X is codinates (normalized) of a picture, 
		# colorData_train is RGB value that is normalized and matched coordinates on the picture	
		input_to_model = {X: coor_train, Y_: colorData_train
						,lr: learning_rate}
		_, color_predicted, correct, error = sess.run(
							[optimizer, ColorPredicted, accuracy, loss], input_to_model)
	
		# color_predicted size: [high*width] x 3, 1 row per RGB value		
		return 	color_predicted, correct, error
	
	return trainModel # return inner function

#############################################################	
# ************Test on jpg and png format only****************
if __name__ == '__main__':	
	MAX_STEP = 10000 # all training step
	PATH_PIC_OUPUT = "output.jpg" 		# output file
	PATH_PIC_INPUT = "chicken-test.jpg"	# input file

    # get datasets (Correct answer)
	imageData =  ut.getAllImageData(PATH_PIC_INPUT)
	# size of image data: high x width x RGB
	high, width, _ = imageData.shape
	# These values will feed into the neural network model
	coordTrain = ut.getCoordTrain(high, width)
	colorTrain = ut.getColorDataInPixel(imageData) # target (Correct answer)
	assert coordTrain.shape[0] == colorTrain.shape[0]
	
	trainModel = getTrainModel() # get train function
	
	# Viaualize the image that AI are creating
	ut.visualize(imageData, PATH_PIC_OUPUT, trainModel, coordTrain, colorTrain, MAX_STEP)