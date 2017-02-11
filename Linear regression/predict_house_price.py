from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import util as ut

def show_result(w0, w1, mse):
	print('\nf(x) = %s + %sx , and MSE = %s' % (w0, w1, mse))
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
	fx = X * C							# Linear line for prediction
	mse = mean_squared_error(Y, fx )
	w0, w1 = C
	show_result(w0, w1, mse)	

# Method 2: use sklearn module
def train_method2(X, Y):
	regr = linear_model.LinearRegression()
	regr.fit(X, Y)
	#Finish training
	
	#Show model
	fx = regr.predict(X)				# Linear line for prediction
	mse = mean_squared_error(Y, fx )
	show_result(regr.intercept_, regr.coef_, mse)
	
# Method 3: use numpy module (polyfit)
def train_method3(X, Y):
	X = X.reshape(-1)
	Y = Y.reshape(-1)
	w1, w0 = np.polyfit(X, Y, 1)
	#Finish training
	
	#Show model
	# use  broadcasting rules in numpy to add matrix
	fx = w1*X + [w0]					# Linear line for prediction
	mse = mean_squared_error(Y, fx )
	show_result(w0, w1, mse)
		
# Method 4: use gradient descent algorithm
def isConvergence(value):				# check condition of convergence
	return np.absolute(value) <= 0.01  	# set threshold

def plot_contour_error(data_X, Y): 		# for visualization
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

# Method 4: 	
def train_method4(data_X, Y):
	X = add_one(data_X)	
	learningRate = 0.0001				# initial learning rate
	C = np.matrix([0, 0]).T				# initial coefficients
	
	fx_init = X * C							
	mse = mean_squared_error(Y, fx_init)
	print('\nFirst: f(x) = %s + %sx , and MSE = %s' % (C[0,0] , C[1,0] , mse))

	while(True):		
		slope = X.T * ( X * C - Y) 		# vector 2 x 1
		new_C = C - (learningRate * slope)		# vector 2 x 1
	
		w0, w1 = ( C[0,0], C[1,0] )
		s0, s1 = ( slope[0,0], slope[1,0] )
		
		if isConvergence(s0) == False:
			w0 = new_C[0,0]				# new w0
				
		if isConvergence(s1) == False:
			w1 = new_C[1,0]				# new w1
	
		C = np.matrix([ w0, w1]).T		# update new coefficients
		
		# stop while_loop when w0 and w1 meet convergence condition
		if isConvergence(s0) and isConvergence(s1): 
			break
	#Finish training
	
	#Show model
	fx_final = X * C							# Linear line for prediction
	mse = mean_squared_error(Y, fx_final )
	w0, w1 = C
	show_result(w0, w1, mse)
	
	# for visualization
	plt.plot(data_X, Y, 'bs', data_X, fx_final, 'r-')
	plt.show()

# method5: use tensorflow library
def train_method5(data_X, data_Y):
	# Try to find values for W_1 and W_0 that compute Y = W_1 * data_X + W_0
	W_1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
	W_0 = tf.Variable(tf.zeros([1]))
	Y = W_1 * data_X + W_0

	# Minimize the mean squared errors.
	loss = tf.reduce_mean(tf.square(Y - data_Y))
	
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

#######################	
## for test only	
if __name__ == '__main__':
	# เดี่ยวไปหา dataset ชุดใหม่
	train_XList, train_Y = ut.prepare_dataset('dataset.csv' 
							,x_column_names=['Input'], y_column_name='Output')
	train_X = train_XList[0]
	
	print("\n+++++Show method 1++++")
	train_method1(train_X, train_Y)

	print("\n+++++Show method 2++++")	
	C_model = train_method2(train_X, train_Y)
	
	print("\n+++++Show method 3++++")
	train_method3(train_X, train_Y)
	
	plot_contour_error(train_X, train_Y)
	print("\n+++++Show method 4++++")
	train_method4(train_X, train_Y)
	
	print("\n+++++Show method 5++++")
	train_method5(train_X, train_Y)		
