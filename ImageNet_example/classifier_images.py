import numpy as np
import matplotlib.pyplot as plt

"""
Models for image classification with weights trained on ImageNet:
* Xception
* VGG16
* VGG19
* ResNet50
* InceptionV3

ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), 
in which each node of the hierarchy is depicted by hundreds and thousands of images. 
(Cite: www.image-net.org/)

Peper reference:
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/abs/1610.02357 (Xception model)

Very Deep Convolutional Networks for Large-Scale Image Recognition
https://arxiv.org/abs/1409.1556 (VGG models)

Deep Residual Learning for Image Recognition
https://arxiv.org/abs/1512.03385 ( ResNet model)

Rethinking the Inception Architecture for Computer Vision
https://arxiv.org/abs/1512.00567 ( Inception v3 model )
"""

# imagenet class index: 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
# credit library: https://keras.io/applications
# source code of library: https://github.com/fchollet/deep-learning-models
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.models import Model

def prepareImage(img, modelName):
	# dim_ordering == 'th': backend is Theano , 	chanel x heigh x width
	# dim_ordering == 'tf': backend is TensorFlow, 	heigh x width x chanel
	imgData = image.img_to_array(img)
	
	# image shape (th): [1, chanel, width, height]
	# image shape (tf): [1, width, height, chanel]	
	imgData = np.expand_dims(imgData, axis=0)
	
	# select function: preprocess_input for VGG16, VGG19, ResNet50 model (same functions)
	# Convert 'RGB'->'BGR'
	# and then subtract color mean values (BGR): [123.68, 116.779, 103.939]
	if modelName == 'VGG16':
		from keras.applications.vgg16 import preprocess_input
	elif modelName == 'VGG19':
		from keras.applications.vgg19 import preprocess_input
	elif modelName == 'ResNet50':
		from keras.applications.resnet50 import preprocess_input
		
	# select function: preprocess_input for InceptionV3 and Xception model (same functions)
	# imgData /= 255.
	# imgData -= 0.5
	# imgData *= 2.
	elif modelName == 'InceptionV3':
		from keras.applications.inception_v3 import preprocess_input
	elif modelName == 'Xception':
		from keras.applications.xception import preprocess_input
	else:
		raise ValueError
		
	imgData = preprocess_input(imgData)
	print('Input image shape:', imgData.shape)
	return imgData

def _deprocessImg(imgData):
	# an interval of [0, 255] is specified, values smaller than 0 become 0, 
	# and values larger than 255 become 255.
	return np.clip(imgData, 0, 255).astype(np.uint8)
	
def showPredict(orgImg, predicted):
	className = [val[1] for val in predicted[0]]
	prop = [val[2] for val in predicted[0]]
	
	f, (ax1, ax2) = plt.subplots(2)
	ax1.set_title('Resize the image: %s' % str(np.shape(orgImg)) )	
	ax1.imshow(orgImg)
	ax1.set_axis_off()
	
	N = 5
	ind = np.arange(N)  # the x locations for the groups
	width = 0.35       # the width of the bars
	
	ax2.set_ylabel('Probability')
	ax2.set_title('This is a -> %s' % className[0], color='red')
	ax2.set_ylim([0, 1.3])
	ax2.get_xaxis().set_visible(False)
	rects1 = ax2.bar(ind, prop, width, color='blue')	
	
	# add some text for labels and title 
	for index, rect in enumerate(rects1):
		height = rect.get_height()
		col = 'black'
		if index == 0:
			col = 'red'
		ax2.text(rect.get_x() + rect.get_width()/2., 1.05*height,
				'%s\n(%.2f)' % (className[index], prop[index]),
				ha='center', va='bottom', color=col)
		
	plt.gcf().canvas.set_window_title('Image classification')
	plt.show()

def getPreds_top5(model, imgData):
	preds = model.predict(imgData)
	preds_top5 = decode_predictions(preds, top=5)
	print('Predicted:', preds_top5)
	return preds_top5

def visualizeModel(base_model, imgData, block_name):
	# Never been tested for ResNet50 InceptionV3 and Xception model
	"""
	Structure: VGG16 model
	# Block 1:  			'block1_conv1', 'block1_conv2', 'block1_pool'
	# Block 2:	   			'block2_conv1', 'block2_conv2', 'block2_pool'
	# Block 3: 				'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_pool'
	# Block 4:				'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_pool'
	# Block 5:				'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_pool'
	# Classification block: 'fc1', 'fc2', 'predictions'
	
	Structure: VGG19 model
	# Block 1:  			'block1_conv1', 'block1_conv2', 'block1_pool'
	# Block 2:	   			'block2_conv1', 'block2_conv2', 'block2_pool'
	# Block 3: 				'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_conv4', 'block3_pool'
	# Block 4:				'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4', 'block4_pool'
	# Block 5:				'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4', 'block5_pool'
	# Classification block: 'fc1', 'fc2', 'predictions'
	"""
	
	feature_model = Model(input=base_model.input, output=base_model.get_layer(block_name).output)
	features = feature_model.predict(imgData)
	# [1, heigh, width, filter]	
	features = features[0] 
	
	_, _, totalFilter = features.shape
	features = features.transpose(2,0,1)
		
	f, (axList) = plt.subplots(2,4)
	axList = np.reshape(axList, (-1,))
	randIndex = np.random.randint(0, totalFilter, size=8)
	
	for index, numFilter in enumerate(randIndex):
		ax = axList[index]
		ax.set_axis_off()
		ax.set_title('Filter : %s' % numFilter)
		plt.gray()
		ax.imshow(_deprocessImg(features[numFilter]))
		ax.set_axis_off()
	
	plt.gcf().canvas.set_window_title('Visualize VGG model')	
	plt.show()

# test here
if __name__ == '__main__':
	print('\n+++++Example 1: Image classification with VGG16 model++++++')
	# The default input size for VGG16 model is 224x224.
	img = image.load_img('monkey.jpg', target_size=(224, 224))
	imgData = prepareImage(img, 'VGG16')
	
	"""
	TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5'
	TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
	TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
	TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
	"""
	model = VGG16(include_top=True, weights='imagenet')
	preds_top5 = getPreds_top5(model, imgData)	
	showPredict(img, preds_top5)
	
	print('\nWaiting visualize....')
	visualizeModel(model, imgData, block_name='block1_pool')
	
	print('\n+++++Example 2: Image classification with VGG19 model++++++')
	# The default input size for VGG19 model is 224x224.
	img = image.load_img('tiger.jpg', target_size=(224, 224))
	imgData = prepareImage(img, 'VGG19')
	"""
	TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_th_dim_ordering_th_kernels.h5'
	TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
	TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_th_dim_ordering_th_kernels_notop.h5'
	TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
	"""
	model = VGG19(include_top=True, weights='imagenet')
	preds_top5 = getPreds_top5(model, imgData)
	showPredict(img, preds_top5)
	
	print('\nWaiting visualize....')
	visualizeModel(model, imgData, block_name='block1_pool')
	
	print('\n+++++Example 3: Image classification with ResNet50 model++++++')
	# The default input size for ResNet50 model is 224x224.
	img = image.load_img('bicycle.jpg', target_size=(224, 224))
	imgData = prepareImage(img, 'ResNet50')
	"""
	TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5'
	TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
	TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5'
	TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'	
	"""	
	model = ResNet50(include_top=True, weights='imagenet')
	preds_top5 = getPreds_top5(model, imgData)
	showPredict(img, preds_top5)
	
	print('\n+++++Example 4: Image classification with InceptionV3 model++++++')
	#The default input size for InceptionV3 model is 299x299.
	img = image.load_img('bird.jpg', target_size=(299, 299))
	imgData = prepareImage(img, 'InceptionV3')	
	"""
	TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/inception_v3_weights_th_dim_ordering_th_kernels.h5'
	TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
	TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/inception_v3_weights_th_dim_ordering_th_kernels_notop.h5'
	TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
	"""
	model = InceptionV3(include_top=True, weights='imagenet')
	preds_top5 = getPreds_top5(model, imgData)
	showPredict(img, preds_top5)
	
	print('\n+++++Example 5: Image classification with Xception model++++++')
	# Xception model is only available for the TensorFlow backend, 
	# The default input size for Xception model is 299x299.
	img = image.load_img('rabbit.jpg', target_size=(299, 299))
	imgData = prepareImage(img, 'Xception')	
	"""
	TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'
	TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
	"""
	model = Xception(include_top=True, weights='imagenet')
	preds_top5 = getPreds_top5(model, imgData)
	showPredict(img, preds_top5)
	