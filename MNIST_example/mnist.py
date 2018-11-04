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
import time
import datetime
import numpy as np

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import SimpleRNN, LSTM, GRU 
from keras.models import Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D

#np.random.seed(137)  # for reproducibility
LABEL = np.arange(10) # [0, 1, 2, 3, 4, 5, 6, 7 ,8 ,9]
def encode(Y):	
	# for example: 0 will to be encoded [1, 0, 0, 0, 0, 0 ,0, 0, 0, 0]
	# In python, True is 1, False = 0	
	Yencoded = np.array( [ 1*(LABEL == digit) for digit in Y ])
	assert Yencoded.shape[1] == len(LABEL) # 10
	return Yencoded
	
def decode(Ydigits):
	# np.argmax: Returns the indices of the maximum values along an axis.	
	return np.array( [ np.argmax(list) for list in Ydigits] )
	
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
	X_image = X.reshape((-1, W, W))
	return X_image
	
def plotExampleImg(title, X_image, Ydigits, Y_predict=None):
	fig, axarr = plt.subplots(2, 5)
	axList = np.reshape(axarr, (2*5,))
	plt.gcf().canvas.set_window_title(title)
	fig.set_facecolor('#FFFFFF')
	assert X_image.shape[0] == Ydigits.shape[0]
		
	for num in range(0,10):	 # label 0 to 9
		selectIndex = np.where(Ydigits == num)[0] # select all indexs followed the label number
		digitsImg = X_image[selectIndex]		
		# random images
		#Return random integers from 0 (inclusive) to high (exclusive).
		randomIndex = np.random.randint(0, digitsImg.shape[0])		
		#axList[num].imshow(digitsImg[randomIndex], cmap=plt.cm.gray)	
		plt.gray()
		axList[num].set_axis_off() # turn off axis x, y
		axList[num].imshow(digitsImg[randomIndex])	
		if Y_predict is not None:
			assert Ydigits.shape[0] == Y_predict.shape[0]
			ySelect = Y_predict[selectIndex]
			axList[num].set_title("%s=> (%.2f)" % (num, ySelect[randomIndex]))		
		else:			
			axList[num].set_title("Number %s" % num)
		
	plt.tight_layout()
	plt.show()
	
def plotPCA2d(title, X_train, Ydigits):
	estimator = PCA(n_components=2)
	Xpca = estimator.fit_transform(X_train)	
	colors = ['red', 'green','blue', 'black', 'purple', 'pink', 'orange', 'gray', 'violet', 'olive']	
	for number in range(0, 10): # 0 to 9
		select_index = np.where(Ydigits == number)[0]
		XY = Xpca[select_index]
		# seperate to x, y component
		x = XY[:, 0]	
		y = XY[:, 1]		
		plt.scatter(x, y, c=colors[number])
	plt.title(title)
	plt.legend(np.arange(0,10), loc='upper right')	
	plt.xlabel('First Principal Component')
	plt.ylabel('Second Principal Component')
	plt.show()

# Example 1: Nearest neighbors	
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
	print("Accuracy: %.4f" % accuracy)
	print("Classification report")
	print(metrics.classification_report(Yexpected, Ypredicted))
	
# Example 2: Support vector
def train_support_vector(Xtrain, Ytrain, Xtest, Yexpected):
	# a support vector classifier
	classifier = svm.SVC(gamma=0.001)
	# learning
	classifier.fit(Xtrain, Ytrain)
	# predict
	Ypredicted = classifier.predict(Xtest)
	
	# Calculate accuracy (True in python is 1, and False is 0
	accuracy = np.sum(Yexpected == Ypredicted)/ len(Yexpected) * 100
	print("Accuracy: %.4f" % accuracy)
	print("Classification report")
	print(metrics.classification_report(Yexpected, Ypredicted))

# For remain examples
def trainModel(model, Xtrain, Ytrain, Xtest, Yexpected, epochs):
	global_start_time = time.time()
	#automatic validation dataset 
	model.fit(Xtrain, Ytrain, batch_size=500, epochs=epochs, verbose=0, validation_data=(Xtest, Yexpected))		
	#model.fit(Xtrain, Ytrain, batch_size=500, nb_epoch=epochs, verbose=0, validation_split=0.8)
	sec = datetime.timedelta(seconds=int(time.time() - global_start_time))
	print ('Training duration : ', str(sec))
	
	# evaluate all training set after trained
	scores = model.evaluate(Xtrain, Ytrain, verbose=0)
	print("Evalute model: %s = %.4f" % (model.metrics_names[0] ,scores[0]))
	print("Evalute model: %s = %.4f" % (model.metrics_names[1] ,scores[1]*100))	

	# for test only
	scores = model.evaluate(Xtest, Yexpected, verbose=0)
	print("Test model: %s = %.4f" % (model.metrics_names[0] ,scores[0]))
	print("Test model: %s = %.4f %%" % (model.metrics_names[1] ,scores[1]*100))
	return model
	
def testModel(model, X_image, Xtest, Yexpected,  title_graph=""):
	Ypredicted 			= model.predict(Xtest, verbose=0)
	#Yexpected 			= decode(Yexpected) 	# convert binary to digits 0-9
	Ypredicted_decode 	= decode(Ypredicted) # convert binary to digits 0-9
	print("Classification report")
	print(metrics.classification_report(Yexpected, Ypredicted_decode))	
	Y_max = np.array( [ np.max(list) for list in Ypredicted] )
	plotExampleImg(title_graph, X_image, Ypredicted_decode, Y_max)
	
# Example 3: Logistic regression (1 neural)
def build_logistic_regression(features):
	model = Sequential()		
	# L2 is weight regularization penalty, also known as weight decay, or Ridge
	model.add(Dense(input_dim=features, output_dim=10, W_regularizer=l2(0.20))) 
	# now model.output_shape == (None, 10)
	# note: `None` is the batch dimension.	
	#
	model.add(Activation("softmax"))
		
	# algorithim to train models use RMSprop
	# compute loss with function: categorical crossentropy
	model.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])
	return model

# Example 4: Multilayer Perceptron (MLP)
def build_MLP(features):		
	model = Sequential()
	model.add(Dense(input_dim=features, units=500))
	# now model.output_shape == (None, 500)
	# note: `None` is the batch dimension.
	#
	model.add(Activation("relu"))
	model.add(Dropout(0.6))	# reduce overfitting
	#
	model.add(Dense(10))
	#now model.output_shape == (None, 10)	
	model.add(Activation("softmax")) #outputs are independent 
		
	# algorithim to train models use RMSprop
	# compute loss with function: categorical crossentropy
	model.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])	
	return model

# For example 5
def reshapeCNN2D_Input(X): 
	exampleNum, D = X.shape	
	W = int(math.sqrt(D))	
	assert W == 8 # size of image == 8 x 8
	
	# change shape of image data	 			
	if K.image_dim_ordering() == 'th': 
		# backend is Theano
		# Image dimension = chanel x row x column (chanel = 1, if it is RGB: chanel = 3)
		XImg = X.reshape(exampleNum, 1, W, W)			
	else: 
		# 'tf' backend is Tensorflow
		# Image dimension = row x column x chanel (chanel = 1, if it is RGB: chanel = 3)
		XImg = X.reshape(exampleNum, W, W, 1)				
		
	return XImg

# For example 6
def reshapeCNN1D_Input(X): 
	exampleNum, D = X.shape	
	W = int(math.sqrt(D))	
	assert W == 8 # size of image == 8 x 8	
	return X.reshape(exampleNum, W, W)			

# Example 5: Convolutional Neural Networks (CNN) with Conv2D
def build_CNN_2D(image_shape):	
	model = Sequential()
	# apply a 3x3 convolution (filter 2D) with 100 output filters on a 8 x 8 image:
	# Theano: image data size is chanel x row x colum (1 x 8 x 8)
	# Tensorflow: image data size is row x colum x chanel (8 x 8 x 1)
	model.add(Conv2D(filters=100, kernel_size=(3, 3), padding='same',
                        input_shape=image_shape))
	# now model.output_shape == (None, 100, 8, 8)
	# note: `None` is the batch dimension.
	#
	model.add(Conv2D(filters=100, kernel_size=(3, 3), padding='same'))
	# now model.output_shape(None, 100, 8, 8)
	#
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	# now model.output_shape == (None, 100, 4, 4)
	#
	model.add(Dropout(0.5))	# reduce overfitting
	#
	model.add(Flatten())	
	# now model.output_shape == (None, 100 x 4 x 4)
	#
	model.add(Dense(10))
	model.add(Activation('softmax')) 
	# now model.output_shape == (None, 10)

	# algorithim to train models use ADAdelta
	# compute loss with function: categorical crossentropy
	model.compile(optimizer='adadelta',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])
	return model

# Example 6: Convolutional Neural Networks (CNN) with Convolution1D
def build_CNN_1D(image_shape):	 
	model = Sequential()
	# apply a 3 convolution (filter 1D) with 100 output filters on a 8 x 8 image:
	model.add(Conv1D(filters=100, kernel_size= 3, padding='same', input_shape=image_shape))
	# now model.output_shape == (None, 100, 8)
	# note: `None` is the batch dimension.
	#
	model.add(Conv1D(filters=100, kernel_size=3, padding='same'))
	# now model.output_shape(None, 100, 8)
	#
	model.add(Activation('relu'))
	model.add(MaxPooling1D(pool_size=2))
	# now model.output_shape == (None, 100, 4)
	#
	model.add(Dropout(0.5))	# reduce overfitting
	#
	model.add(Flatten())	
	# now model.output_shape == (None, 100 x 4)
	#
	model.add(Dense(10))
	model.add(Activation('softmax')) 
	# now model.output_shape == (None, 10)

	# algorithim to train models use ADAdelta
	# compute loss with function: categorical crossentropy
	model.compile(optimizer='adadelta',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])
	return model
			  
# For example 7, 8, 9 ==> for Recurrent Neural Networks
def getSequenceInput(X): 
	exampleNum, D = X.shape	
	W = int(math.sqrt(D))	
	assert W == 8 # size of image == 8 x 8
	
	# Dimension = row x colum (without chanel)
	XImg = X.reshape(exampleNum, W, W)
	return XImg

# Example 7: Recurrent Neural Networks (RNNs)
def build_RNN(image_shape):			
	sequence, features = image_shape
	model = Sequential()
	# apply a LSM with 8 sequences (row) and 8 features (column) on a 8 x 8 image:
	model.add(SimpleRNN(	input_shape=(sequence, features),				
					units=200, dropout=0.2, recurrent_dropout=0.2, 	return_sequences=True))
	# now model.output_shape == (None, 200)
	# note: `None` is the batch dimension.
	#
	model.add(SimpleRNN(200, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
	# now model.output_shape == (None, 200)	
	#
	model.add(Dense(10))
	#now model.output_shape == (None, 10)	
	model.add(Activation("softmax")) #outputs are independent 
		
	# algorithim to train models use RMSProp
	# compute loss with function: categorical crossentropy
	model.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])	
	return model
	
# Example 8: Long short-term memory (LSTM)
def build_LSTM(image_shape):		
	sequence, features = image_shape
	model = Sequential()
	# apply a LSM with 8 sequences (row) and 8 features (column) on a 8 x 8 image:
	model.add(LSTM(	input_shape=(sequence, features),				
					units=200, dropout=0.2, recurrent_dropout=0.2, 	return_sequences=True))
	# now model.output_shape == (None, 200)
	# note: `None` is the batch dimension.
	#
	model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
	# now model.output_shape == (None, 200)	
	#
	model.add(Dense(10))
	#now model.output_shape == (None, 10)	
	model.add(Activation("softmax")) #outputs are independent 
		
	# algorithim to train models use RMSProp
	# compute loss with function: categorical crossentropy
	model.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])	
	return model

# Example 9: Gated Recurrent Unit (GRU)
def build_GRU(image_shape):			
	sequence, features = image_shape
	model = Sequential()
	# apply a LSM with 8 sequences (row) and 8 features (column) on a 8 x 8 image:
	model.add(GRU(	input_shape=(sequence, features),				
					units=200, dropout=0.2, recurrent_dropout=0.2, 	return_sequences=True))
	# now model.output_shape == (None, 200)
	# note: `None` is the batch dimension.
	#
	model.add(GRU(200, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
	# now model.output_shape == (None, 200)	
	#
	model.add(Dense(10))
	#now model.output_shape == (None, 10)	
	model.add(Activation("softmax")) #outputs are independent 
		
	# algorithim to train models use RMSProp
	# compute loss with function: categorical crossentropy
	model.compile(optimizer='rmsprop',
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])	
	return model