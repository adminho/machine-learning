from mnist import getDatasets, restoreImg, plotExampleImg, plotPCA2d, encode

# For example 1 and 2
from mnist import train_nearest_neighbors, train_support_vector

from mnist import trainModel, testModel
from mnist import build_logistic_regression, build_MLP 		# For example 3 and 4
from mnist import build_CNN_2D, reshapeCNN2D_Input 			# For example 5
from mnist import build_CNN_1D, reshapeCNN1D_Input 			# For example 6
from mnist import getSequenceInput, build_RNN, build_LSTM, build_GRU # For example 7, 8 and 9

if __name__ == "__main__":
	Xtrain, Xtest, Ytrain, Ytest = getDatasets()
	assert Xtrain.shape[0] == Ytrain.shape[0]	# number of samples
	assert Xtrain.shape[1] == 64   				# total pixel per a image
	print("Size of training input:", Xtrain.shape)
	print("Size of testing input:", Xtest.shape)
	
	X_image = restoreImg(Xtrain)	
	X_testImage = restoreImg(Xtest)	
	plotExampleImg("Show image examples", X_image, Ytrain)
	plotPCA2d("Show PCA MNIST", Xtrain, Ytrain)
		
	print("\n+++++ Example 1: Nearest neighbors ++++")
	train_nearest_neighbors(Xtrain, Ytrain, Xtest, Ytest)
	
	print("\n+++++ Example 2: Support vector ++++")
	train_support_vector(Xtrain, Ytrain, Xtest, Ytest)	
	
	#number of examples, features (8x8)
	_, features = Xtrain.shape
	YtrainEncoded = encode(Ytrain) 	# transform labels format to binary digits
	YtestEncoded = encode(Ytest)	# transform labels format to binary digits
	assert YtrainEncoded.shape[0] == Ytrain.shape[0]
	assert YtestEncoded.shape[0] == Ytest.shape[0]
	
	print("\n+++++ Example 3: Logistic regression ++++")
	model = build_logistic_regression(features)	
	model = trainModel(model, Xtrain, YtrainEncoded, Xtest, YtestEncoded, epochs=200)	
	testModel(model, X_testImage, Xtest, Ytest, title_graph="Example 3: Logistic regression")
		
	print("\n+++++ Example 4: Multilayer Perceptron (MLP) ++++")
	model = build_MLP(features)
	model = trainModel(model, Xtrain, YtrainEncoded, Xtest, YtestEncoded, epochs=50)	
	testModel(model, X_testImage, Xtest, Ytest, title_graph="Example 4: Multilayer Perceptron (MLP)")
	
	print("\n+++++ Example 5: Convolutional neural network (CNN) with Convolution2D ++++")
	print("Take a minute.....")		
	# reshape to Theano: (batchsize, chanel, row, colum) or Tensorflow: (batchsize, row, column, chanel)
	XtrainCNN = reshapeCNN2D_Input(Xtrain)
	XtestCNN = reshapeCNN2D_Input(Xtest)
	image_shape = XtrainCNN.shape[1:]	# select (chanel, row, column) or (row, column, chanel)
	model = build_CNN_2D(image_shape)
	model = trainModel(model, XtrainCNN, YtrainEncoded, XtestCNN, YtestEncoded, epochs=50)
	testModel(model, X_testImage, XtestCNN, Ytest, title_graph="Example 5: Convolutional neural network (CNN) with Convolution2D)")
		
	print("\n+++++ Example 6: Convolutional neural network (CNN) with Convolution1D ++++")
	print("Take a minute.....")		
	# reshape to Theano: (batchsize, row, colum) without chanel
	XtrainCNN = reshapeCNN1D_Input(Xtrain)
	XtestCNN = reshapeCNN1D_Input(Xtest)
	image_shape = XtrainCNN.shape[1:]	# select (row, column)
	model = build_CNN_1D(image_shape)
	model = trainModel(model, XtrainCNN, YtrainEncoded, XtestCNN, YtestEncoded, epochs=50)
	testModel(model, X_testImage, XtestCNN, Ytest, title_graph="Example 6: Convolutional neural network (CNN) with Convolution1D")
	
	# reshape to sequences for Recurrent Neural Networks
	XtrainSeq = getSequenceInput(Xtrain)
	XtestSeq = getSequenceInput(Xtest)
	image_shape = XtrainSeq.shape[1:]	# select (row, column)
	
	print("\n+++++ Example 7: Recurrent Neural Networks (RNN) ++++")
	print("Take a minute.....")	
	model = build_RNN(image_shape)
	model = trainModel(model, XtrainSeq, YtrainEncoded, XtestSeq, YtestEncoded, epochs=30)
	testModel(model, X_testImage, XtestSeq, Ytest, title_graph="Example 7: Recurrent Neural Networks (RNN)")
	
	print("\n+++++ Example 8: Long short-term memory (LSTM) ++++")
	print("Take a minute.....")		
	model = build_LSTM(image_shape)
	model = trainModel(model, XtrainSeq, YtrainEncoded, XtestSeq, YtestEncoded, epochs=30)
	testModel(model, X_testImage, XtestSeq, Ytest, title_graph="Example 8: Long short-term memory (LSTM)")

	print("\n+++++ Example 9: Gated Recurrent Unit (GRU) ++++")
	print("Take a minute.....")		
	model = build_GRU(image_shape)
	model = trainModel(model, XtrainSeq, YtrainEncoded, XtestSeq, YtestEncoded, epochs=30)	
	testModel(model, X_testImage, XtestSeq, Ytest, title_graph="Example 9: Gated Recurrent Unit (GRU)")