from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import join
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress
import tensorflow as tf
import animation as am		

from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.models import Sequential

def show_result(w0, w1, mse):
	print('f(x) = %s + %sX , and MSE = %s' % (w0, w1, mse))
	print('Coefficients are w0 = %s, w1 = %s' % (w0, w1))
		
def add_one(data_X):
	ones =np.ones((len(data_X),1))	
	# Add 1 vector to frist column
	X = np.append(ones, data_X, axis=1)	 
	# Wrap matrix to X, then you can use operators of the matrix
	X = np.matrix(X)
	return X	
	
# Method 1: solve eqation
def train_method1(data_X, Y):
	X = add_one(data_X)	
	C = (X.T * X).I * (X.T * Y) 		# Answer of coefficients 	
	#Finish training
	
	#Show model
	FX = X * C							# Linear line for prediction
	mse = mean_squared_error(Y, FX )
	w0, w1 = C
	show_result(w0, w1, mse)	

# Method 2: use sklearn module
def train_method2(X, Y):
	regr = linear_model.LinearRegression()
	regr.fit(X, Y)
	#Finish training
	
	#Show model
	FX = regr.predict(X)				# Linear line for prediction
	mse = mean_squared_error(Y, FX )
	show_result(regr.intercept_, regr.coef_, mse)
	
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
	
# Method 6: use gradient descent algorithm (hard code manual)
def isConvergence(value):				# check condition of convergence
	return np.absolute(value) <= 0.01  	# set threshold

def plot_surface_error(data_X, Y): 		# for visualization
	x_range = np.arange(-5,15,1)		# -5< x-axis < 15 (increase 1 step)
	y_range = np.arange(-5,15,1)		# -5< y-axis < 15 (increase 1 step)

	w0, w1 = np.meshgrid(x_range, y_range)
	Z = np.empty(w0.shape) 	# same size as w0 and w1
	X = add_one(data_X)	
	
	for i in x_range: 		# calculate mse of all (w0, w1) and save to Z
		for j in y_range: 
			C = np.matrix( [w0[i, j] , w1[i, j]]).T			
			fx = X * C
			Z[i,j] = mean_squared_error(Y, fx )	 		

	fig = plt.figure() 
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('wo axis'); ax.set_ylabel('w1 axis'); ax.set_zlabel('MSE axis')  
	surf = ax.plot_surface(w0, w1, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)		
	#ax.zaxis.set_major_locator(LinearLocator(10))
	#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	plt.show()

def isNan(value):
	if np.sum( np.isnan(value)) > 0 :
		return True

# Method 4: 	
def train_method7(data_X, Y):
	X = add_one(data_X)	
	learningRate = 0.0001				# initial learning rate
	C = np.matrix([0, 0]).T				# initial coefficients
	
	FX_init = X * C							
	mse_init = mean_squared_error(Y, FX_init)
	print('First: f(x) = %s + %sx , and MSE = %s' % (C[0,0] , C[1,0] , mse_init))
	
	# save predicted price for visualization later
	FX_List = [FX_init]		
	step = 0
	
	while(True):		
		SLOPE = X.T * ( X * C - Y) 		# vector 2 x 1
		new_C = C - (learningRate * SLOPE)		# vector 2 x 1
				
		if isNan(SLOPE):
			print('Slope is NaN:', SLOPE)
			break
			
		w0, w1 = C[0,0], C[1,0]
		s0, s1 = SLOPE[0,0], SLOPE[1,0]
		
		if isConvergence(s0) == False:
			w0 = new_C[0,0]				# new w0
				
		if isConvergence(s1) == False:
			w1 = new_C[1,0]				# new w1
		
		C = np.matrix([ w0, w1]).T		# update new coefficients
		
		if step % 100 == 0: # for visualization later
			FX = X * C			
			FX_List = np.append(FX_List, FX) 
		step +=1
		
		# stop while_loop when w0 and w1 meet convergence condition
		if isConvergence(s0) and isConvergence(s1): 
			break
	#Finish training
	print("Total step to learning:", step)
	
	#Show model
	FX_final = X * C							
	mse_final = mean_squared_error(Y, FX_final )
	w0, w1 = C
	show_result(w0, w1, mse_final)
	
	# for visualization
	FX_List = np.append(FX_List, FX_final) 
	FX_List = np.reshape(FX_List,(-1, X.shape[0]))  # number of fx values x number of DatasetX
	return FX_List	
	
def prepare_dataset(csv_dataset,x_column_name, y_column_name, base_dir  = "" ):
	# read csv file with pandas module	
	df = pd.read_csv(join(base_dir, csv_dataset))
	
	print("First of 5 row in Dataset")
	print(df.head())	
	print("\nTail of 5 row in Dataset")
	print(df.tail())
	
	train_X = df[x_column_name].reshape(-1,1)		# X (Input) training set
	train_Y = df[y_column_name].reshape(-1,1)			# Y (Output) training set	
	return train_X, train_Y
	
#######################	
## for test only
def run_testsuite(train_X, train_Y):		
	print("\nSize of training set X: {}".format(train_X.shape))
	print("Size of training set Y: {}".format(train_Y.shape))

	print("\n+++++Show method 1++++")
	train_method1(train_X, train_Y)
	
	print("\n+++++Show method 2++++")	
	train_method2(train_X, train_Y)
	
	print("\n+++++Show method 3++++")
	train_method3(train_X, train_Y)
	
	print("\n+++++Show method 4++++")
	train_method4(train_X, train_Y)		

	print("\n+++++Show method 5++++")
	train_method5(train_X, train_Y)
	
	print("\n+++++Show method 6++++")
	#train_method6(train_X, train_Y)
	
	# uncomment this if you want to show contour of error graph
	#plot_surface_error(train_X, train_Y)
	print("\n+++++Show method 7++++")
	FX_List = train_method7(train_X, train_Y)
	am.visualize(train_X, train_Y, FX_List)	
		
if __name__ == '__main__':	
	train_X, train_Y = prepare_dataset(csv_dataset='food_truck.csv'
								,x_column_name='population'
								,y_column_name='profit')
	run_testsuite(train_X, train_Y)	
 
	train_X, train_Y = prepare_dataset(csv_dataset='example_price_house_40_headcolumn.csv'
								,x_column_name='area'
								,y_column_name='price')
	scaler = StandardScaler().fit(train_X)
	train_X_new = scaler.transform(train_X)	 # Normalized

	run_testsuite(train_X_new, train_Y)	
	