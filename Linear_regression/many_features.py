from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.datasets import load_boston

from scipy.stats import linregress
import tensorflow as tf

from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.models import Sequential

def show_result(w0, w_remain, mse):
	print('Bias  = %s , Coefficients = %s , MSE = %s' % (w0, w_remain, mse))

def show_graph(X, Y, predict, title, xlabel, ylabel):
	plt.scatter(X, Y, color='b', label='data')
	plt.plot(X, predict, color='r', label='predict')
	plt.title('Thailand population')
	plt.xlabel('Years')	
	plt.ylabel('Population')
	plt.show()	

def add_one(data_X):
	ones =np.ones((len(data_X),1))	
	# Add 1 vector to frist column
	X = np.append(ones, data_X, axis=1)	 
	# Wrap matrix to X, then you can use operators of the matrix
	X = np.matrix(X)
	return X	
	
# Method 1: solve eqation
def predict_example1(data_X, Y):
	X = add_one(data_X)	
	C = (X.T * X).I * (X.T * Y) 			# Answer of coefficients 	
	#Finish training
	
	#Show model
	predict = X * C							# nonlinear line for prediction
	mse = mean_squared_error(Y, predict )
	w0, w_remain = C[0], C[1:]
	show_result(w0, w_remain, mse)	
	return predict

# Method 2: use sklearn module
def predict_example2(X, Y):
	regr = linear_model.LinearRegression()
	regr.fit(X, Y)
	#Finish training
	
	#Show model
	predict = regr.predict(X)				# nonlinear line for prediction
	mse = mean_squared_error(Y, predict)
	show_result(regr.intercept_, regr.coef_, mse)
	return predict
	
	
# method 5: use tensorflow library
def predict_example3(X, Y):
	# Try to find values for weights and bias that compute FX = W * X + B	
	num_feature = X.shape[1]
	W = tf.Variable(tf.truncated_normal((num_feature, 1)))
	B = tf.Variable(tf.truncated_normal((1, 1)))	
	X = X.astype(np.float32)	
	FX =tf.matmul(X, W) + B	
	
	# Minimize the mean squared errors.
	loss = tf.reduce_mean(tf.square(FX - Y))
	
	# Use gradient descent algorithm for optimizing
	learningRate = 0.01
	optimizer = tf.train.GradientDescentOptimizer(learningRate)	
	train = optimizer.minimize(loss)

	# Before starting, initialize the variables.  We will 'run' this first.
	init = tf.global_variables_initializer()

	# Launch the graph.
	sess = tf.Session()
	sess.run(init)

	# Try fit the line
	b, w, mse = (None, None, None)
	for step in range(3000):
		_, b, w, mse, fx = sess.run([train, B, W, loss, FX])				
		
	#Show model
	show_result(b, w ,mse )

# method 6: use Keras library
def predict_example4(X, Y):
	model = Sequential()
	num_feature = X.shape[1]	
	model.add(Dense(1, input_dim=num_feature, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01))
	weights = model.layers[0].get_weights()	
	model.fit(X, Y, epochs=3000, verbose=0)
	
	weights = model.layers[0].get_weights()
	w = weights[0]
	b = weights[1][0]
	
	#Show model
	FX = model.predict(X)				# Linear line for prediction
	mse = mean_squared_error(Y, FX)
	show_result(b, w, mse)

def prepare_dataset(csv_dataset,x_column_name, y_column_name, base_dir  = "" ):
	# read csv file with pandas module	
	df = pd.read_csv(join(base_dir, csv_dataset))
	
	print("First of 5 row in Dataset")
	print(df.head())	
	print("\nTail of 5 row in Dataset")
	print(df.tail())
	
	train_X = df[x_column_name].values.reshape(-1,1)		# X (Input) training set
	train_Y = df[y_column_name].values.reshape(-1,1)		# Y (Output) training set	
	return train_X, train_Y

#######################	
## for test only
def test_one_input(X, train_Y ,title, xlabel, ylabel):	
	# Preprocessing data
	scaler_X = preprocessing.StandardScaler().fit(X)
	scaler_Y = preprocessing.StandardScaler().fit(train_Y)
	X__ = scaler_X.transform(X)
	train_Y = scaler_Y.transform(train_Y)	
	#print(X__[0:5])
	#print(train_Y[0:5])
	
	# Generate polynomial features (2 degree).
	degree = 2 # You can change to degree 3, 4, 5 ant other at here
	poly = PolynomialFeatures(degree)	
	# output for degree 2 = [1, x, x^2]
	# output for degree 3 = [1, x, x^2, x^3]
	train_X = poly.fit_transform(X__)
	#print(train_X[0:5])
	
	# remove 1 value from array (index 0)
	train_X = np.delete(train_X, [0], 1)	
	#print(train_X[0:5])
	
	assert train_X.shape == (len(train_X), degree)  # (xxx, degree)
	assert train_Y.shape == (len(train_Y), 1) 		# (xxx, 1)
		
	print("\n+++++ Example 1++++")
	predict = predict_example1(train_X, train_Y)
	show_graph(X, 
			scaler_Y.inverse_transform(train_Y), 
			scaler_Y.inverse_transform(predict) 
			,title, xlabel, ylabel) 	

	print("\n+++++ Example 2++++")	
	predict = predict_example2(train_X, train_Y)
		
	print("\n+++++ Example 3++++")
	predict = predict_example3(train_X, train_Y)
	
	print("\n+++++ Example 4++++")
	predict_example4(train_X, train_Y)		

def test_many_input(train_X, train_Y):	
	scaler_X = preprocessing.StandardScaler().fit(train_X)
	scaler_Y = preprocessing.StandardScaler().fit(train_Y)
	train_X = scaler_X.transform(train_X)
	train_Y = scaler_Y.transform(train_Y)
	
	print("\n+++++ Example 1++++")
	predict = predict_example1(train_X, train_Y)
	
	print("\n+++++ Example 2++++")	
	predict = predict_example2(train_X, train_Y)
	
	print("\n+++++ Example 3++++")	
	predict = predict_example3(train_X, train_Y)
	
	print("\n+++++ Example 4++++")	
	predict = predict_example4(train_X, train_Y)
	
if __name__ == '__main__':
	print("------- One input but many features (polynomial features) --------")
	print("++++++++ Example: Thailand population history ++++++++")
	X, train_Y = prepare_dataset('Thailand_population_history.csv'
									,x_column_name='Year'
									,y_column_name='Population')
	test_one_input(X, train_Y, title='Thailand population', xlabel='Years', ylabel='Population')
	
	print("\n+++++++++++++++++++++++++++++++++++++++++++++")	
	print("++++++++ Example: Average income per month per household (B.E 41-58) ++++++++")
	X, train_Y = prepare_dataset('average_income_per_month_per_household_41-58.csv'
									,x_column_name='Years'
									,y_column_name='Average Monthly Income Per Household')
	test_one_input(X, train_Y, title='Thailand monthly income', xlabel='Years', ylabel='Average Monthly Income Per Household')

	print("\n++++++++++++ Many input ++++++++++++")
	print("++++++++++++ Boston datasets +++++++++")
	boston = load_boston()
	train_X, train_Y = boston.data, boston.target	
	train_Y = np.reshape(train_Y, (len(train_Y),1)) # shape: len(train_Y) x 1	
	test_many_input(train_X, train_Y)
	