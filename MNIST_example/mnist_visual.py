# reference code example from: https://github.com/martin-gorner/tensorflow-mnist-tutorial

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Model

import matplotlib.animation as animation
import math
import datetime, time

# my modules
from mnist import getDatasets, restoreImg, plotExampleImg, encode, testModel
from mnist import build_logistic_regression
from history import TrainingHistory
from visual import Visualization		

def pad(image, x_pad=2, y_pad=1):		
	# pad zero to image
	#z1 = np.full((image.shape[0], x_pad) ,0 ,dtype='uint8')	
	z1 = np.zeros((image.shape[0], x_pad))
	# pad zero to left of image
	padding = np.append(z1, image, axis=1)
	# pad zero to right of image
	padding = np.append(padding, z1, axis=1)
	
	#z2 = np.full((y_pad, padding.shape[1]) ,0 ,dtype='uint8')
	z2 = np.zeros((y_pad, padding.shape[1]))
	# pad zero to up of image	
	padding = np.vstack([z2, padding])
	# pad zero to low of image
	padding = np.vstack([padding, z2])		
	return padding
	
def random10Image(Xtrain, Ydigits): # random 10 images
	randImage = []
	for num in range(0,10):			
		digitsImg = Xtrain[np.where(Ydigits == num)[0]]
		#Return random integers from 0 (inclusive) to high (exclusive).
		randomIndex = np.random.randint(0, digitsImg.shape[0])		
		#axList[num].imshow(digitsImg[randomIndex], cmap=plt.cm.gray)
		randImage.append(digitsImg[randomIndex])
	return np.array(randImage)

def preprocess(image):	
	return np.clip(10 *image, 0 ,255) # increase color but limit between 0 - 255
	
def combineImage(randImage): # combine 10 images to 1 image
	randImage = restoreImg(randImage)	
	randImage =[ pad(preprocess(img)) for img in randImage ]	
	img_top = randImage[0]
	for i in range(1,5):
		img_top = np.append(img_top, randImage[i], axis=1)	
	img_upper = randImage[5]
	for i in range(6,10):		
		img_upper = np.append(img_upper, randImage[i], axis=1)
	return  np.append(img_top, img_upper, axis=0)

def build_neural_network(features):		
	model = Sequential()
	model.add(Dense(input_dim=features, units=400))
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
	print(model.summary())
	return model
	
def getHiddenLayer(model):
	input = model.inputs[0]		
	output = model.layers[2].output
	#print(model.layers)
	return Model(input, output)

if __name__ == "__main__":
	Xtrain, Xtest, Ytrain, Ytest = getDatasets()
	his = TrainingHistory()
	_, features = Xtrain.shape
	model = build_neural_network(features)

	YtrainEncoded = encode(Ytrain) 				# transform labels format to binary digits
	Yexpected = encode(Ytest)

	rand10Image = random10Image(Xtrain, Ytrain) # select 10 images
	packedImage = combineImage(rand10Image)		# combine 10 images to 1 images
	hiddenModel = getHiddenLayer(model)

	# call this function in a loop to train the model
	def training_model(num_epochs=10, step_visual=0, visual=None): 	
		model.fit(Xtrain, YtrainEncoded, batch_size=500, epochs=num_epochs, verbose=0, callbacks=[his], validation_data=(Xtest, Yexpected))	 	
		if visual is None:	
			return 
		
		# for visualisation 	
		outputHidden = hiddenModel.predict(rand10Image)	 # predict 10 images
		packedImage2 = combineImage(outputHidden)
		#probability = [predict[i, i] for i in range(0, 10)]
		#visual.update_predict(probability)
		visual.update_accuracy_line(his.accuracy, his.val_accuracy)
		visual.update_loss_line(his.loss, his.val_loss)	
		visual.update_image(packedImage)
		visual.update_image_hidden( packedImage2 )		
		print("\n============ step %s ============" % (step_visual*num_epochs) )
		print("Training: accuracy = %s and loss = %s" % (his.accuracy[-1:], his.loss[-1:]))
		print("Validation: accuracy = %s and loss = %s\n" % (his.val_accuracy[-1:], his.val_loss[-1:]))
	
	visual = Visualization(title="MNIST example with neural network")
	visual.train(training_model, iterations=30, save_movie=False)
	
	X_testImage = restoreImg(Xtest)	
	testModel(model, X_testImage, Xtest, Ytest, title_graph="Example 4: Multilayer Perceptron (MLP)")