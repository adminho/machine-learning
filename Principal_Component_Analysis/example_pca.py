# Reference: http://adataanalyst.com/scikit-learn/principal-component-analysis-scikit-learn/
import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def getDatasets():
	digits = load_digits()
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


def plotPCA2d(Xpca, Ydigits):
	colors = ['red', 'green','blue', 'black', 'purple', 'yellow', 'orange', 'gray', 'lime', 'cyan']
	for number in range(0, 10): # 0 to 9
		XY = Xpca[np.where(Ydigits == number)[0]]
		# seperate to x, y component
		x = XY[:, 0]	
		y = XY[:, 1]
		plt.scatter(x, y, c=colors[number])
	plt.legend(np.arange(0,10))
	plt.xlabel('First Principal Component')
	plt.ylabel('Second Principal Component')
	plt.show()
	
def plotPCA3d(Xpca, Ydigits):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	colors = ['red', 'green','blue', 'black', 'purple', 'yellow', 'orange', 'gray', 'lime', 'cyan']
	for number in range(0, 10): # 0 to 9
		XYZ = Xpca[np.where(Ydigits == number)[0]]
		# seperate to x, y component
		x = XYZ[:, 0]	
		y = XYZ[:, 1]
		z = XYZ[:, 1]
		ax.scatter(x, y, z, c=colors[number])
	
	ax.legend(np.arange(0,10))
	ax.set_xlabel('First Principal Component')
	ax.set_ylabel('Second Principal Component')
	ax.set_zlabel('Third Principal Component')
	plt.show()

def getPCAvalues(X, n_components):
	estimator = PCA(n_components=n_components)
	return estimator.fit_transform(X)	
	
if __name__ == "__main__":
	Xtrain, _, Ycorrect,_ = getDatasets()
	assert Xtrain.shape[0] == Ycorrect.shape[0]		# number of samples
	assert Xtrain.shape[1] == 64   					# total pixel per a image

	imageData = restoreImg(Xtrain)
	plotExampleImg("Show example:", imageData, Ycorrect)	# for test only

	# convert 64 components to 2 components
	Xpca = getPCAvalues(Xtrain, 2)
	assert Xpca.shape[0] == Ycorrect.shape[0]
	assert Xpca.shape[1] == 2

	plotPCA2d(Xpca, Ycorrect)
	
		# convert 64 components to 3 components
	Xpca = getPCAvalues(Xtrain, 3)
	assert Xpca.shape[0] == Ycorrect.shape[0]
	assert Xpca.shape[1] == 3
	plotPCA3d(Xpca, Ycorrect)

