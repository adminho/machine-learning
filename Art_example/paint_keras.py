# Inspiration: http://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html
import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

import util as ut

def getTrainModel(): # The model at here	 
	chanel = 3 # Red, Green, Blue in a pixel	
	
	# fully connected layers but last layer has 3 neurons or nodes (3 chanel)
	K = 20  # number of neurons in first fully connected layer
	L = 20  # number of neurons in second fully connected layer
	M = 20  # number of neurons in third fully connected layer
	N = 20  # number of neurons in forth fully connected layer
	O = 20  # number of neurons in fifth fully connected layer
	P = 20  # number of neurons in sixth fully connected layer
	Q = 20	# number of neurons in seventh fully connected layer

	model = Sequential()
	# model.input_shape == (None, 2)
	# model.output_shape == (None, K)
	# note: `None` is the batch dimension.
	model.add(Dense(input_dim=2, units=K, activation='relu'))
	model.add(Dense(units=L, activation='relu'))	
	model.add(Dense(units=M, activation='relu'))	
	model.add(Dense(units=N, activation='relu'))	
	model.add(Dense(units=O, activation='relu'))		
	model.add(Dense(units=P, activation='relu'))		
	model.add(Dense(units=Q, activation='relu'))		
	model.add(Dense(units=3))	 # (None, Red, Green, Blue)

	optm=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)		
	# algorithim to train models use
	# compute loss with function: mse ( mean squared error)
	model.compile(optimizer=optm,
			  loss='mse',
			  metrics=['accuracy'])	
		
	def trainModel(step, coor_train, colorData_train):
		# coor_train is codinates (normalized) of a picture, 
		# colorData_train is RGB value that is normalized and matched coordinates on the picture	
		# color_predicted size: [high*width] x 3, 1 row per RGB value		
		
		model.fit(coor_train, colorData_train, epochs=1,  verbose=0)
		color_predicted = model.predict(coor_train)
		scores = model.evaluate(coor_train, colorData_train, verbose=0)
		correct = scores[0]
		error = scores[1]		
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