# Inspiration: http://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html
import os
import numpy as np
import scipy.misc
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

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
	# Loss function, I use MSE function
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
	
def getAllImageData(path):
	img = scipy.misc.imread(path).astype(np.float)
	if len(img.shape) == 2: # grayscale		
		img = np.dstack((img,img,img))	
	return img

def getColorDataInPixel(imageData):			
	imageData = np.array(imageData)
	high, width, _ = imageData.shape
	
	# reshape size of image data to size: [high*width] x 3
	# 1 row per color information (RGB) in a pixel	
	image_train = imageData.reshape(high*width, -1)

	_, chanel = image_train.shape	
	if chanel >= 4: # for .png file (png format is RGBA, jpg format is RGB)						
		image_train = np.delete(image_train[:], 3, axis=1) # select RGB only
	
	return image_train/255 # normalized		

def getCoordTrain(high, width):		
	coordinates = np.zeros((high, width, 2))		
	for w in range(0,width): # get all coordinates (x,y)
		for h in range(0,high):			 
			coordinates[h][w][0] = (w-width/2)/width; # normalize x codinate			 
			coordinates[h][w][1] = (h-high/2)/high;   # normalize y codinate
	
	# reshape coordinates to size: [high*width] x 2
	coordinates  = np.array(coordinates)
	coord_train = coordinates.reshape(high*width , -1)
	return coord_train

def preShowImage(imageData):
	return np.clip(imageData, 0, 255).astype(np.uint8)
	
def restoreImage(colPredict, high, width):	
	# restore RGB values from normalized		
	colPredict = np.floor(255*colPredict)	
	# Restore shape of color_predicted to size: high x width x 3	
	imageData = colPredict.reshape((high, width , 3) )
	return imageData

# get datasets (Correct answer)
imageData =  getAllImageData('chicken-test.jpg')
# size of image data: high x width x RGB
high, width, _ = imageData.shape

# Show the image that created
fig = plt.figure()
plt.gcf().canvas.set_window_title("Drawing")
fig.set_facecolor('#FFFFFF')
ax1 = fig.add_subplot(1,2,1)
ax1.grid(False) # toggle grid off
ax1.set_axis_off()
ax1.set_title('Orginal Picture')

ax2 = fig.add_subplot(1,2,2)
ax2.grid(False) # toggle grid off
ax2.set_axis_off()
ax2.set_title('AI Painting')

originImg = preShowImage(imageData)
orginImax = ax1.imshow(originImg, animated=True, cmap='binary', vmin=0.0, vmax=1.0, interpolation='nearest', aspect=1.0)
showPicBegin = np.ones(imageData.shape)
paintImax = ax2.imshow(showPicBegin, animated=True, cmap='binary', vmin=0.0, vmax=1.0, interpolation='nearest', aspect=1.0)        

# These values will feed into the neural network model
coord_train = getCoordTrain(high, width)
color_train = getColorDataInPixel(imageData)
train = getTrainModel() # get train function

#saver = tf.train.Saver([W1,W2,W3,W4,W5,W6,W7,W8,B1,B2,B3,B4,B5,B6,B7,B8])	# save your model					
def initImg_func():
	return orginImax, paintImax

def showImage(color_predicted):	
	imgData = restoreImage(color_predicted, high, width)
	imgShow = preShowImage(imgData)
	paintImax.set_data(imgShow)	# draw a predicted image
	return imgShow

import sys
MAX_STEP = 10000 # all training step
PATH_OUPUT = "output.jpg"
def updateImg_func(step):
	# Train your model (each step)
	colPredict, correct, error = train(step, coord_train, color_train)			
	if correct > 99 or step == MAX_STEP-1 :		
		print('Finised at step %d | loss %f  | accuracy %f' % (step, error, correct))
		imgShow = showImage(colPredict) # visualize a image that AI is painting
		#ani.event_source.stop()	   # stop show image
		#saver.save(sess, saved_wieght_file, global_step=step)
		scipy.misc.imsave(PATH_OUPUT, imgShow)
		return orginImax, paintImax
			
	if step %10 == 0:
		print('step %d | loss %f  | accuracy %f' % (step, error, correct))
		showImage(colPredict)	# visualize a image that AI is painting	
			
	return orginImax, paintImax

# ************Test on jpg and png format only	
def main(): 
	ani = FuncAnimation(fig, updateImg_func, frames=np.arange(0, MAX_STEP),
                    init_func=initImg_func, repeat=False, blit=True)		
	plt.show()

if __name__ == '__main__':	
    main()