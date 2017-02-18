import os
import numpy as np
import scipy.io
import math
from matplotlib import pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split

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
		axList[num].imshow(numberImg[randomIndex])
	
	plt.axis('off')
	plt.show()
	
def example1():	
	# Other example: iris = fetch_mldata('iris', data_home=test_data_home)
	# test_data_home is directory for saving mat file
	mnist = datasets.fetch_mldata("MNIST Original")	
	x = mnist.data
	y = mnist.target
	print(mnist.keys()) # dict_keys(['COL_NAMES', 'data', 'target', 'DESCR'])
	
	x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.33,
                                                        random_state=42)
		
	_, D = x_train.shape	
	W = int(math.sqrt(D))	
	assert D == 784 and W == 28
		
	imageData = x_train.reshape((-1, W, W))	# picture 28 x 28
	plotExampleImg("Example: 1", imageData, y_train)

from sklearn.datasets import load_digits
def example2():
	digits = load_digits()
	x = digits.data
	y = digits.target
	
	images = digits.images
	_, h, w = images.shape
	assert h == 8 and w == 8				# picture 8 x 8
	print(digits.keys())
	
	x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.33,
                                                        random_state=42)
														
	_, D = x_train.shape	
	W = int(math.sqrt(D))	
	assert D == 64 and W == 8
	
	imageData = x_train.reshape((-1, W, W))	# picture 8 x 8
	plotExampleImg("Example: 2", imageData, y_train)														

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
def example3():
	# mnist.test : 10K images + labels
	# mnist.train : 60K images + labels
	# 'data' is the directory that save all datasets
	mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
	# batches of 100 images with 100 labels
	batch_X, batch_Y = mnist.train.next_batch(100)
	assert batch_X.shape == (100, 28, 28, 1)
	assert batch_Y.shape == (100, 10)
	
	# Y label is encoded : for example 9 is [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
	batch_temp = np.argsort(batch_Y)
	# decoded
	batch_label = np.array([val[9] for val in batch_temp])
	assert batch_label.shape == (100,)
	
	batch_X = batch_X.reshape(-1,28 , 28)	# 100 x 28 x 28
	plotExampleImg("Example: 3", batch_X, batch_label)

if __name__ == "__main__":
	example1()
	example2()
	example3()