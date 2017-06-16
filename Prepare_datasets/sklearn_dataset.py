import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reference: http://scikit-learn.org/stable/datasets/index.html (14/06/2017)
"""
scikit-learn comes with a few small standard datasets that do not require to download any file from some external website.
load_boston([return_X_y])	Load and return the boston house-prices dataset (regression).
load_iris([return_X_y])	Load and return the iris dataset (classification).
load_diabetes([return_X_y])	Load and return the diabetes dataset (regression).
load_digits([n_class, return_X_y])	Load and return the digits dataset (classification).
load_linnerud([return_X_y])	Load and return the linnerud dataset (multivariate regression).
"""

##### Simple Images ##### 
# The scikit also embed a couple of sample JPEG images published under Creative Commons license by their authors. Those image can be useful to test algorithms and pipeline on 2D data.
# load_sample_images()	Load sample images for image manipulation.
# load_sample_image(image_name)	Load the numpy array of a single sample image
from sklearn.datasets import load_sample_images
dataset = load_sample_images()
"""
Parameters:	
image_name: {`china.jpg`, `flower.jpg`} :
The name of the sample image loaded
Returns:	
img: 3D array :
The image as a numpy array: height x width x color
"""     
print("++++++ Sample Images ++++++\n")
img_data = dataset.images
filenames = dataset.filenames
#print(dataset.DESCR)
print("Shape :", np.shape(img_data))
fig = plt.figure()
plt.gcf().canvas.set_window_title("Sample Image")
for position in range (1, len(img_data)+1):
	ax = fig.add_subplot(2, 1, position)
	ax.set_axis_off()
	ax.set_title(filenames[position-1])
	ax.imshow(img_data[position-1])	
plt.show()
print()


##### Load and return the boston house-prices dataset (regression) ##### 
from sklearn.datasets import load_boston
boston = load_boston()
"""
Parameters:	
return_X_y : boolean, default=False.
If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object.
New in version 0.18.
Returns:	
data : Bunch
Dictionary-like object, the interesting attributes are: ‘data’, the data to learn, ‘target’, the regression targets, and ‘DESCR’, the full description of the dataset.
(data, target) : tuple if return_X_y is True
New in version 0.18.
"""
print("++++++ Boston house-prices dataset (regression) ++++++\n")
#print(boston.DESCR)
print("Shape X:", boston.data.shape)
print("Shape y:", boston.target.shape)
column_head = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE" , "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
df =pd.DataFrame(columns=column_head, data=boston.data)
df["MEDV"] = boston.target
print(df.head())
print()


##### Load and return the iris dataset (classification) #####
from sklearn.datasets import load_iris
data = load_iris()
"""
Parameters:	
return_X_y : boolean, default=False.
If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object.
New in version 0.18.
Returns:	
data : Bunch
Dictionary-like object, the interesting attributes are: ‘data’, the data to learn, ‘target’, the classification labels, ‘target_names’, the meaning of the labels, ‘feature_names’, the meaning of the features, and ‘DESCR’, the full description of the dataset.
(data, target) : tuple if return_X_y is True
New in version 0.18.
"""
print("++++++ Load and return the iris dataset (classification) ++++++\n")
#print(data.DESCR)
# The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width.
print("Shape X:", data.data.shape)
print("Shape y:", data.target.shape)
# 3 different types of irises ['setosa' 'versicolor' 'virginica']
print("Class name:", data.target_names) 
print()


#### Load and return the diabetes dataset (regression) ####
from sklearn.datasets import load_diabetes
data = load_diabetes()
"""
Parameters:	
return_X_y : boolean, default=False.
If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object.
New in version 0.18.
Returns:	
data : Bunch
Dictionary-like object, the interesting attributes are: ‘data’, the data to learn and ‘target’, the regression target for each sample.
(data, target) : tuple if return_X_y is True
New in version 0.18.
"""
print("++++++ Load and return the diabetes dataset (regression) ++++++\n")
print("Shape X:", data.data.shape)
print("Shape y:", data.target.shape)
print()


#### Load and return the digits dataset (classification) ####
from sklearn.datasets import load_digits
digits = load_digits()
"""
Parameters:	
n_class : integer, between 0 and 10, optional (default=10)
The number of classes to return.
return_X_y : boolean, default=False.
If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object.
New in version 0.18.
Returns:	
data : Bunch
Dictionary-like object, the interesting attributes are: ‘data’, the data to learn, ‘images’, the images corresponding to each sample, ‘target’, the classification labels for each sample, ‘target_names’, the meaning of the labels, and ‘DESCR’, the full description of the dataset.
(data, target) : tuple if return_X_y is True
New in version 0.18.
"""
print("++++++ Load and return the digits dataset (classification) ++++++\n")
#print(digits.DESCR)
print("Shape X:", digits.images.shape)
print("Shape y:", digits.target.shape)
print()
fig = plt.figure()
plt.gcf().canvas.set_window_title("Digits dataset")
axList = []
for position in range (1,11):
		ax = fig.add_subplot(2,5,position)
		ax.set_axis_off()
		axList.append(ax)	
for num in range(0,10):	
		ax = axList[num]						
		ax.set_title("Label: %d" % num)
		index_list = np.where(digits.target == num)[0]
		selected_imgList = digits.images[index_list]
		random_index = np.random.randint(0, selected_imgList.shape[0])
		ax.imshow(selected_imgList[random_index])			
plt.gray()
plt.show()
print()


#### Load and return the linnerud dataset (multivariate regression) ####
from sklearn.datasets import load_linnerud
data = load_linnerud()

"""
Parameters:	
return_X_y : boolean, default=False.
If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object.
New in version 0.18.
Returns:	
data : Bunch
Dictionary-like object, the interesting attributes are: ‘data’ and ‘targets’, the two multivariate datasets, with ‘data’ corresponding to the exercise and ‘targets’ corresponding to the physiological measurements, as well as ‘feature_names’ and ‘target_names’.
(data, target) : tuple if return_X_y is True
New in version 0.18.
"""
print("++++++ Load and return the linnerud dataset (multivariate regression) ++++++\n")
print(digits.DESCR)
print("Shape X:", data.data.shape)
print("Shape y:", data.target.shape)
print("feature_names:", data.feature_names)
print("target_names:", data.target_names)
print()