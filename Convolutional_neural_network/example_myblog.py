# Convolution example easy
import numpy as np
image   = [ [1, 1, 1, 1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 5, 1, 1, 1, 1],
			[1, 1, 1, 1, 5, 1, 1, 1, 1],
			[1, 1, 1, 1, 5, 1, 1, 1, 1],
			[1, 5, 5, 5, 5, 5, 5, 5, 1],
			[1, 1, 1, 1, 5, 1, 1, 1, 1],
			[1, 1, 1, 1, 5, 1, 1, 1, 1],
			[1, 1, 1, 1, 5, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1, 1, 1, 1]]

image2  = [ [1, 1, 1, 1, 1, 1, 1, 1, 1],
			[1, 5, 1, 1, 1, 1, 1, 5, 1],
			[1, 1, 5, 1, 1, 1, 5, 1, 1],
			[1, 1, 1, 5, 1, 5, 1, 1, 1],
			[1, 1, 5, 1, 5, 1, 1, 1, 1],
			[1, 1, 1, 5, 1, 5, 1, 1, 1],
			[1, 1, 5, 1, 1, 1, 5, 1, 1],
			[1, 5, 1, 1, 1, 1, 1, 5, 1],
			[1, 1, 1, 1, 1, 1, 1, 1, 1]]

			
filter1 =  [[-1,  1, -1],
			[ 1,  1,  1],
			[-1,  1, -1]]
			
filter2 =  [[-2,  2, -2],
			[-2,  2, -2],
			[-2,  2, -2]]

			
filter3 =  [[-3, -3, -3],
			[ 3,  3,  3],
			[-3, -3, -3]]
	

def pad_zeros(image, filter):	
	size_row, size_col = filter.shape		
	begin_row = int((size_row-1)/2)
	end_col = int((size_col-1)/2)
	
	# pad zero to image
	z1 = np.zeros((image.shape[0], end_col))
	# pad zero to left of image
	padding = np.append(z1, image, axis=1)
	# pad zero to right of image
	padding = np.append(padding, z1, axis=1)
	
	z2 = np.zeros((begin_row, padding.shape[1]))	
	# pad zero to up of image	
	padding = np.vstack([z2, padding])
	# pad zero to low of image
	padding = np.vstack([padding, z2])		
	return padding

def scan(image, filter, pad, to_do):	
	image = np.array(image)
	filter = np.array(filter)

	limit_row, limit_col = filter.shape	
	length_row, length_col = 0, 0	
	if pad == "same":
		image = pad_zeros(image, filter)
	elif pad != "valid":
		raise Exception("Invalid pad parameter")
	
	length_row = image.shape[0]-limit_row+1
	length_col = image.shape[1]-limit_col+1		
	feature_map = []
	for row in range(0, length_row):
		for col in range(0, length_col):
			# crop image followed by filter size
			crop = image[row: row+limit_row, col: col+limit_col]			
			# convolution or polling(sub sample)
			value = to_do(crop, filter)
			feature_map.append(value)
	
	# reshape list to hight x width
	feature_map = np.reshape(feature_map, (length_row, length_col))
	return feature_map
	
	
def convolve(image, filter, pad):	
	return scan(image, filter, pad, lambda crop, filter: np.sum(crop * filter))

def max_polling(feature_map, size_hight, size_width, pad="valid"):	
	filter = np.zeros((size_hight, size_width))
	# resue this function same as convolve
	return scan(feature_map, filter, pad, lambda crop, filter: np.max(crop))

def relu(feature_map):
	return np.maximum(feature_map, 0)

image = image2
print("|||||||||Image|||||||||||\n") 
print(np.array(image))

print("\n+++++++Convolution without padding++++++++")
print(convolve(image, filter1,  pad="valid"))

print("\n+++++++++Convolution with padding+++++++")
c1 = convolve(image, filter1,  pad="same")
c2 = convolve(image, filter2,  pad="same")
c3 = convolve(image, filter3,  pad="same")
print("Feature map1 with filter 1") 
print(c1)
print("\nFeature map2 with filter 2\n")
print(c2)
print("\nFeature map3 with filter 3\n")
print(c3)


print("\n+++++++++Max polling layer+++++++")
p1 = max_polling(c1, 3, 3)
p2 = max_polling(c2, 3, 3)
p3 = max_polling(c3, 3, 3)
print("Feature map1 with max polling") 
print(p1)
print("\nFeature map2 with max polling")
print(p2)
print("\nFeature map3 with max polling")
print(p3)

print("\n+++++++++ReLU activation +++++++++++++++++")
relu1 = relu(p1)
relu2 = relu(p2)
relu3 = relu(p3)
print("Feature map1 with max polling") 
print(relu1)
print("\nFeature map2 with max polling")
print(relu2)
print("\nFeature map3 with max polling")
print(relu3)

print("\n+++++++++Flatten +++++++++++++++++")
# concatenate 
X = np.array( [ relu1, relu2, relu3] )
# reshpape to 1 dimension
X = X.reshape(-1, )
print("Length of vector:", np.shape(X))

print("\n+++++++++Fully connected layer +++++++++++++++++")
"""
def sigmoid(X):
	return 1/(1+np.exp(-X))

import pickle
def save_myweight(X):
	# I assume weights (parameters) of neural network
	W = 10*np.matrix(X).I

	pkl_file = open('weights_example.pkl', 'wb')
	pickle.dump(W, pkl_file)
	pkl_file.close()
	
#save_myweight(X)

weights_file = open('weights_example.pkl', 'rb')
W = pickle.load(weights_file)
weights_file.close()

# predict image finally
predict = np.matmul(X, W)
print("Finally predict:")
print(sigmoid(predict))	
"""
