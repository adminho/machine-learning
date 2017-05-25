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

from scipy.stats import linregress
import tensorflow as tf

from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.models import Sequential

def show_result(w0, w_remain, mse):
	print('Coefficients are w0 = %s and other w = %s' % (w0, w_remain))

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
	
# Method 3: use numpy module (polyfit)
def train_method3(X, Y):
	X = X.reshape(-1)
	Y = Y.reshape(-1)
	w1, w0 = np.polyfit(X, Y, 1)
	#Finish training
	
	#Show model
	# use  broadcasting rules in numpy to add matrix
	FX = w1*X + [w0]					# Linear line for prediction
	mse = mean_squared_error(Y, FX)
	show_result(w0, w1, mse)

# method 4 : use scipy module(linregress)
def train_method4(X, Y):
	X = X.reshape(1,-1)
	Y = Y.reshape(1,-1)
	slope, intercept, r, p, stderr = linregress(X, Y)
	
	#Show model
	FX = intercept + slope*X				# Linear line for prediction
	mse = mean_squared_error(Y, FX)
	show_result(intercept, slope, mse)
	
# method 5: use tensorflow library
def train_method5(X, Y):
	# Try to find values for W_1 and W_0 that compute FX = W_1 * X + W_0
	W_1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
	W_0 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
	FX = W_1*X + W_0

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
	w0, w1, mse = (0, 0, 0)
	for step in range(3000):
		_, w0, w1, mse = sess.run([train, W_0, W_1, loss])		
			
	#Show model
	show_result(w0, w1, mse)

# method 6: use Keras library
# doesn't work and slowly
def train_method6(X, Y):
	model = Sequential()	
	model.add(Dense(1, input_dim=1, init='normal'))
	model.compile(loss='mean_squared_error', optimizer=Adam(lr=3.5, decay=1.9))
	weights = model.layers[0].get_weights()	
	model.fit(X, Y, nb_epoch=25000, verbose=0)
	
	weights = model.layers[0].get_weights()
	w1 = weights[0][0][0]
	w0 = weights[1][0]
	
	#Show model
	FX = w0 + w1*X				# Linear line for prediction
	mse = mean_squared_error(Y, FX )
	show_result(w0, w1, mse)

	
def prepare_dataset(csv_dataset,x_column_name, y_column_name, base_dir  = "" ):
	# read csv file with pandas module	
	df = pd.read_csv(join(base_dir, csv_dataset))
	
	print("First of 5 row in Dataset")
	print(df.head())	
	print("\nTail of 5 row in Dataset")
	print(df.tail())
	
	train_X = df[x_column_name].reshape(-1,1)		# X (Input) training set
	train_Y = df[y_column_name].reshape(-1,1)		# Y (Output) training set	
	return train_X, train_Y

#######################	
## for test only
def test_one_input(X, train_Y ,title, xlabel, ylabel):
	
	# normalize easily
	FRIST_X = X[0:1]
	FRIST_Y = train_Y[0:1]
	X = X/FRIST_X 					# start at 1 value (divide with frist year)
	train_Y = train_Y/FRIST_Y 		# start value (divide with population of first year)
	#print(X[0:5])
	#print(train_Y[0:5])
	
	# Generate polynomial features (2 degree).
	degree = 2 # You can change to degree 3, 4, 5 ant other at here
	poly = PolynomialFeatures(degree)	
	# output for degree 2 = [1, x, x^2]
	# output for degree 3 = [1, x, x^2, x^3]
	train_X = poly.fit_transform(X)
	#print(train_X[0:5])
	
	# remove 1 value from array (index 0)
	train_X = np.delete(train_X, [0], 1)	
	#print(train_X[0:5])
	
	assert train_X.shape == (len(train_X), degree)  # (xxx, degree)
	assert train_Y.shape == (len(train_Y), 1) 		# (xxx, 1)
		
	print("\n+++++Show example 1++++")
	predict = predict_example1(train_X, train_Y)
	show_graph(X*FRIST_X, train_Y*FRIST_Y, predict*FRIST_Y ,title, xlabel, ylabel) # restore input data before showing graph
	

	print("\n+++++Show example 2++++")	
	predict = predict_example2(train_X, train_Y)
	#show_graph(X*FRIST_X, train_Y*FRIST_Y, predict*FRIST_Y ,title, xlabel, ylabel) # restore input data before showing graph
	
	print("\n+++++Show method 3++++")
	#train_method3(train_X, train_Y)
	
	print("\n+++++Show method 4++++")
	#train_method4(train_X, train_Y)		

	print("\n+++++Show method 5++++")
	#train_method5(train_X, train_Y)

if __name__ == '__main__':	
	X, train_Y = prepare_dataset('Thailand_population_history.csv'
									,x_column_name='Year'
									,y_column_name='Population')
	test_one_input(X, train_Y, title='Thailand population', xlabel='Years', ylabel='Population')
	
	X, train_Y = prepare_dataset('average_income_per_month_per_household_41-58.csv'
									,x_column_name='Years'
									,y_column_name='Average Monthly Income Per Household')
	test_one_input(X, train_Y, title='Thailand monthly income', xlabel='Years', ylabel='Average Monthly Income Per Household')