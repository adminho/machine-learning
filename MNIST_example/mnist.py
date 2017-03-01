""" References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links: [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
Thank you
* http://scikit-learn.org/
* https://keras.io/
"""
import math
import scipy.io
import shutil
import os
import time
import datetime
import numpy as np
import pandas as pd

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2, activity_l2
from keras import backend as K
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#np.random.seed(137)  # for reproducibility
LABEL = np.arange(10) # [0, 1, 2, 3, 4, 5, 6, 7 ,8 ,9]
def encode(Y):	
	# for example 0 will to be encoded [1, 0, 0, 0, 0, 0 ,0, 0, 0, 0]
	# In python, True is 1, False = 0	
	Yencoded = np.array( [ 1*(LABEL == digit) for digit in Y ])
	assert Yencoded.shape[1] == len(LABEL) # 10
	return Yencoded
	
def decode(Ydigits):
	# Returns the indices that would sort an array
	# max values is at last
	return np.array( [ np.argsort(list)[9]	 for list in Ydigits] )
	
def getDatasets():
	digits = datasets.load_digits()
	x = digits.data
	y = digits.target	
	x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.33,
                                                        random_state=42)
	return x_train, x_test, y_train, y_test

def restoreImg(X):
	_, D = X.shape	
	W = int(math.sqrt(D))	
	assert D == W * W
	imageData = X.reshape((-1, W, W))
	return imageData
	
def plotExampleImg(title,imageData, Ydigits):	
	fig = plt.figure()
	plt.gcf().canvas.set_window_title(title)
	fig.set_facecolor('#FFFFFF')
	axList = []
	for position in range (1,11):
		ax = fig.add_subplot(2,5,position)
		ax.set_axis_off()
		axList.append(ax)		
		
	for num in range(0,10):		
		numberImg = imageData[np.where(Ydigits == num)[0]]
		#Return random integers from 0 (inclusive) to high (exclusive).
		randomIndex = np.random.randint(0, numberImg.shape[0])		
		axList[num].imshow(numberImg[randomIndex], cmap=plt.cm.gray)
	
	plt.axis('off')
	plt.show()

# Example 1	
def train_nearest_neighbors(Xtrain, Ytrain, Xtest, Yexpected):
	numSample,_ = Xtest.shape
	# compute distance with Manhattan formula	
	Ypredicted = np.zeros(numSample)	
	for index in range(0,numSample):
		# minus with broadcasting in numpy
		ditanceList = np.sum(np.abs(Xtrain - Xtest[index]), axis=1)		
		# min distance at first of list	and get Index
		minDistanceIndex = np.argsort(ditanceList)[0] 
		Ypredicted[index] = Ytrain[minDistanceIndex]

	# Calculate accuracy (True in python is 1, and False is 0
	accuracy = np.sum(Yexpected == Ypredicted)/ len(Yexpected) * 100
	print("Accuracy %.4f" % accuracy)	
	print("Classification report %s\n" % (metrics.classification_report(Yexpected, Ypredicted)))
	
# Example 2
def train_support_vector(Xtrain, Ytrain, Xtest, Yexpected):
	# a support vector classifier
	classifier = svm.SVC(gamma=0.001)
	# learning
	classifier.fit(Xtrain, Ytrain)
	# predict
	Ypredicted = classifier.predict(Xtest)
	
	# Calculate accuracy (True in python is 1, and False is 0
	accuracy = np.sum(Yexpected == Ypredicted)/ len(Yexpected) * 100
	print("Accuracy %.4f" % accuracy)	
	print("Classification report %s\n" % (metrics.classification_report(Yexpected, Ypredicted)))

# For example 3, 4, 5, 6
def trainModel(model, Xtrain, Ytrain, Xtest, Yexpected, epochs):
	global_start_time = time.time()		
	#automatic validation dataset 
	model.fit(Xtrain, Ytrain, batch_size=500, nb_epoch=epochs, verbose=0, validation_data=(Xtest, Yexpected))			
	sec = datetime.timedelta(seconds=int(time.time() - global_start_time))
	print ('Training duration : ', str(sec))
	
	# evaluate all training set after trained
	scores = model.evaluate(Xtrain, Ytrain, verbose=0)
	print("Evalute model: %s = %.4f" % (model.metrics_names[0] ,scores[0]))
	print("Evalute model: %s = %.4f" % (model.metrics_names[1] ,scores[1]*100))
		
	# for test
	scores = model.evaluate(Xtest, Yexpected, verbose=0)
	print("Test model: %s = %.4f" % (model.metrics_names[0] ,scores[0]))
	print("Test model: %s = %.4f" % (model.metrics_names[1] ,scores[1]*100))
	
# Example 3
def build_logistic_regression(features):
	model = Sequential()		
	# L2 is weight regularization penalty, also known as weight decay, or Ridge
	model.add(Dense(input_dim=features, output_dim=10, W_regularizer=l2(0.20))) 
	# now model.output_shape == (None, 10)
	# note: `None` is the batch dimension.	
	#
	model.add(Activation("sigmoid"))
		
	# algorithim to train models use RMSprop
	# compute loss with function: categorical crossentropy
	model.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])
	return model

# Example 4
def build_neural_network(features):		
	model = Sequential()
	model.add(Dense(input_dim=features, output_dim=500))
	# now model.output_shape == (None, 500)
	# note: `None` is the batch dimension.
	#
	model.add(Activation("relu"))
	model.add(Dropout(0.6))	# reduce overfitting
	#
	model.add(Dense(10))
	#now model.output_shape == (None, 10)	
	model.add(Activation("sigmoid")) #outputs are independent 
		
	# algorithim to train models use RMSprop
	# compute loss with function: categorical crossentropy
	model.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])	
	return model

# For example 5 :Convolutional Neural Networks
def reshapeCNNInput(X): 
	exampleNum, D = X.shape	
	W = int(math.sqrt(D))	
	assert W == 8 # size of image == 8 x 8
	
	# change shape of image data
	input_shape = None 			
	if K.image_dim_ordering() == 'th': 
		# backend is Theano
		# Image dimension = chanel x row x colum (chanel = 1, if it is RGB: chanel = 3)
		XImg = X.reshape(exampleNum, 1, W, W)	
		#input_shape = (1, W, W)
	else: 
		# 'tf' backend is Tensorflow
		# Image dimension = row x colum x chanel (chanel = 1, if it is RGB: chanel = 3)
		XImg = X.reshape(exampleNum, W, W, 1)		
		#input_shape = (W, W, 1)
		
	return XImg

# Example 5
def build_cnn(image_shape):	
	model = Sequential()
	# apply a 3x3 convolution with 80 output filters on a 8 x 8 image:
	# Theano: image data size is chanel x row x colum (1 x 8 x 8)
	# Tensorflow: image data size is row x colum x chanel (8 x 8 x 1)
	model.add(Convolution2D(nb_filter=100, nb_row=3, nb_col=3,
                        border_mode='same',
                        input_shape=image_shape))
	# now model.output_shape == (None, 100, 8, 8)
	# note: `None` is the batch dimension.
	#
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	# now model.output_shape == (None, 100, 4, 4)
	#
	model.add(Dropout(0.5))	# reduce overfitting
	#
	model.add(Flatten())	
	# now model.output_shape == (None, 64x4x4)
	#
	model.add(Dense(10))
	model.add(Activation('sigmoid')) #outputs are independent 

	# algorithim to train models use ADAdelta
	# compute loss with function: categorical crossentropy
	model.compile(optimizer='adadelta',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])
	return model

# For example 6: Long short-term memory
def reshapeLSTMInput(X): 
	exampleNum, D = X.shape	
	W = int(math.sqrt(D))	
	assert W == 8 # size of image == 8 x 8
	
	# Dimension = row x colum
	XImg = X.reshape(exampleNum, W, W)
	return XImg

# Example 6
def build_lstm(image_shape):	
	global_start_time = time.time()	
	sequence, features = image_shape
	model = Sequential()
	# apply a LSM with 8 sequences (row) and 8 features (column) on a 8 x 8 image:
	model.add(LSTM(input_dim=features, 
				input_length=sequence,
				output_dim=150,
				dropout_W=0.2, dropout_U=0.2,
				return_sequences=True))
	# now model.output_shape == (None, 100)
	# note: `None` is the batch dimension.
	#
	model.add(LSTM(150, dropout_W=0.2, dropout_U=0.2, return_sequences=False))
	# now model.output_shape == (None, 100)	
	#
	model.add(Dense(10))
	#now model.output_shape == (None, 10)	
	model.add(Activation("sigmoid")) #outputs are independent 
	
	start = time.time()
	# algorithim to train models use RMSProp
	# compute loss with function: categorical crossentropy
	model.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])	
	return model

if __name__ == "__main__":
	Xtrain, Xtest, Ytrain, Ytest = getDatasets()
	assert Xtrain.shape[0] == Ytrain.shape[0]	# number of samples
	assert Xtrain.shape[1] == 64   				# total pixel per a image
	print("Size of training input:", Xtrain.shape)
	print("Size of testing input:", Xtest.shape)
	
	imageData = restoreImg(Xtrain)
	plotExampleImg("Show example:", imageData, Ytrain)	
	
	print("\n+++++ Nearest neighbors method ++++")
	train_nearest_neighbors(Xtrain, Ytrain, Xtest, Ytest)
	
	print("\n+++++ Support vector method ++++")
	train_support_vector(Xtrain, Ytrain, Xtest, Ytest)	
	
	#number of examples, features (8x8)
	_, features = Xtrain.shape
	YtrainEncoded = encode(Ytrain) 	# transform labels format to digits
	YtestEncoded = encode(Ytest)	# transform labels format to digits
	assert YtrainEncoded.shape[0] == Ytrain.shape[0]
	assert YtestEncoded.shape[0] == Ytest.shape[0]
	
	print("\n+++++ Logistic regression method ++++")
	model = build_logistic_regression(features)	
	trainModel(model, Xtrain, YtrainEncoded, Xtest, YtestEncoded, epochs=200)	
		
	print("\n+++++ Neural network method ++++")
	model = build_neural_network(features)
	trainModel(model, Xtrain, YtrainEncoded, Xtest, YtestEncoded, epochs=50)	
	
	print("\n+++++ Convolutional neural network method ++++")
	# reshape to (batchsize, chanel, row, colum) or (batchsize, row, column, chanel)
	XtrainCNN = reshapeCNNInput(Xtrain)
	XtestCNN = reshapeCNNInput(Xtest)
	image_shape = XtrainCNN.shape[1:]	# select (chanel, row, column) or (row, column, chanel)
	model = build_cnn(image_shape)
	trainModel(model, XtrainCNN, YtrainEncoded, XtestCNN, YtestEncoded, epochs=50)
	
	print("\n+++++ Long short-term memory method ++++")
	print("Take a minute.....")
	XtrainLSTM = reshapeLSTMInput(Xtrain)
	XtestLSTM = reshapeLSTMInput(Xtest)
	image_shape = XtrainLSTM.shape[1:]	# select (row, column)
	model = build_lstm(image_shape)
	trainModel(model, XtrainLSTM, YtrainEncoded, XtestLSTM, YtestEncoded, epochs=30)	