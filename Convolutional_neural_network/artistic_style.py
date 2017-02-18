# Inpiration: https://arxiv.org/pdf/1508.06576v2.pdf (paper)
'''
I borrowed some code  from [Anish Athalye's Neural Style]
@misc{athalye2015neuralstyle,
  author = {Anish Athalye},
  title = {Neural Style},
  year = {2015},
  howpublished = {url{https://github.com/anishathalye/neural-style}},
  note = {commit xxxxxxx}
}

# Another idea from [Justin Johnson]
@misc{Johnson2015,
  author = {Johnson, Justin},
  title = {neural-style},
  year = {2015},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {url{https://github.com/jcjohnson/neural-style}},
}
'''

# My next plan I would like to try implement  
#https://github.com/lengstrom/fast-style-transfer 
#https://github.com/jcjohnson/fast-neural-style 
#http://cs.stanford.edu/people/jcjohns/eccv16/
#https://arxiv.org/abs/1607.08022 (paper)
#https://arxiv.org/abs/1603.08155 (paper)


import tensorflow as tf
import numpy as np
import scipy.io
import time
from matplotlib import pyplot as plt

try:
    reduce
except NameError:
    from functools import reduce

# default arguments
CONTENT_WEIGHT = 5e0
STYLE_WEIGHT = 1e2
TV_WEIGHT = 1e2
LEARNING_RATE = 1e1
STYLE_SCALE = 1.0
ITERATIONS = 1000
DIR_IMAGE_STEP = 'ckeckpoint'

# Use VGG Model: https://arxiv.org/abs/1409.1556 (paper)	
# download : http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'
"""
There are 43 layer in VGG 19 model
layer name		weights(shape)		bias(shape) 
['conv1_1'] 	(3, 3, 3, 64) 		(1, 64)
['relu1_1']
['conv1_2'] 	(3, 3, 64, 64) 		(1, 64)
['relu1_2']
['pool']
['conv2_1'] 	(3, 3, 64, 128) 	(1, 128)
['relu2_1']
['conv2_2'] 	(3, 3, 128, 128) 	(1, 128)
['relu2_2']
['pool']
['conv3_1'] 	(3, 3, 128, 256) 	(1, 256)
['relu3_1']
['conv3_2'] 	(3, 3, 256, 256) 	(1, 256)
['relu3_2']
['conv3_3'] 	(3, 3, 256, 256) 	(1, 256)
['relu3_3']
['conv3_4'] 	(3, 3, 256, 256) 	(1, 256)
['relu3_4']
['pool']
['conv4_1'] 	(3, 3, 256, 512) 	(1, 512)
['relu4_1']
['conv4_2'] 	(3, 3, 512, 512) 	(1, 512)
['relu4_2']
['conv4_3'] 	(3, 3, 512, 512) 	(1, 512)
['relu4_3']
['conv4_4'] 	(3, 3, 512, 512) 	(1, 512)
['relu4_4']
['pool']
['conv5_1'] 	(3, 3, 512, 512) 	(1, 512)
['relu5_1']
['conv5_2'] 	(3, 3, 512, 512) 	(1, 512)
['relu5_2']
['conv5_3'] 	(3, 3, 512, 512) 	(1, 512)
['relu5_3']
['conv5_4'] 	(3, 3, 512, 512) 	(1, 512)
['relu5_4']
['pool']
['fc6'] 		(7, 7, 512, 4096) 		(1, 4096)
['relu6']
['fc7'] 		(1, 1, 4096, 4096) 		(1, 4096)
['relu7']
['fc8'] 		(1, 1, 4096, 1000) 		(1, 1000)
['prob']
"""

# But use 16 conventional layer only
layers = (
	'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
	'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
	'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
	'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
	'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
 )

""" This is model structure for this code
image size: height x weight x chanel or [h, w, c]
but the input into the model has size: [1, h, w, c]

###
layer		 stride		Activation function						 output size
'conv1_1'		1			'relu1_1'********STYLE_LAYERS		[1, h, w ,64]
'conv1_2'		1			'relu1_2'
'pool1'			2		padding='SAME'							[1, h/2, w/2, 64]

'conv2_1'		1			'relu2_1'********STYLE_LAYERS		[1, h/2, w/2 ,128]
'conv2_2'		1			'relu2_2'							[1, h/2, w/2 ,128]
'pool2'			2		padding='SAME'							[1, h/4, w/4 ,128]

'conv3_1'		1			'relu3_1'********STYLE_LAYERS		[1, h/4, w/4 ,256]
'conv3_2'		1			'relu3_2'							[1, h/4, w/4 ,256]
'conv3_3'		1			'relu3_3'							[1, h/4, w/4 ,256]
'conv3_4'		1			'relu3_4'							[1, h/4, w/4 ,256]
'pool3'			2		padding='SAME'							[1, h/8, w/8 ,256]

'conv4_1'		1			'relu4_1'********STYLE_LAYERS		[1, h/8, w/8 ,512]
'conv4_2'		1			'relu4_2'++++++++CONTENT_LAYER		[1, h/8, w/8 ,512]
'conv4_3'		1			'relu4_3'							[1, h/8, w/8 ,512]
'conv4_4'		1			'relu4_4'							[1, h/8, w/8 ,512]
'pool4'			2		padding='SAME'							[1, h/16, w/16 ,512]

'conv5_1'		1			'relu5_1'********STYLE_LAYERS		[1, h/16, w/16 ,512]
'conv5_2'		1			'relu5_2'							[1, h/16, w/16 ,512]
'conv5_3'		1			'relu5_3'							[1, h/16, w/16 ,512]
'conv5_4'		1			'relu5_4'							[1, h/16, w/16 ,512]
"""

def getVGGdata():	
	dataVGG = scipy.io.loadmat(VGG_PATH)	
	dd = dataVGG['layers'][0]
	assert dd.shape == (43,)

	# get color mean
	mean = dataVGG['normalization'][0][0][0]
	assert mean.shape == (224, 224, 3)
	
	# the average color: Red, Green, Blue should be [123.68, 116.779, 103.939])
	meanColor = np.mean(mean, axis=(0, 1))
		
	W={}; B={}
	for index, name in enumerate(layers):
		type = name[:4]
		if type == 'conv':		    
			weights, bias = dd[index][0][0][0][0]
			# mat file	: weights are [width, height, in_channels, out_channels]
			# tensorflow: weights are [height, width, in_channels, out_channels]
			weights = np.transpose(weights, (1, 0, 2, 3))
			# bias : 1 x chanels
			# convert to a vector: chanels x 1		
			bias = bias.reshape(-1)
			W[name] = weights
			B[name] = bias
	
	return W, B, meanColor

W, B, meanColor = getVGGdata()
assert len(W) == 16 # number layer
assert len(B) == 16 # number layer

# Create the network model (can reuse this model)
def createModel(imgInput):
	# The model		
	s1 = (1, 1, 1, 1)
	s2 = (1, 2, 2, 1)
	
	# sytle layer: conv1_1
	# Image will feed into your model later
	conv1_1 = tf.nn.relu(tf.nn.conv2d(imgInput,W['conv1_1'], strides=s1, padding='SAME') + B['conv1_1'])
	conv1_2 = tf.nn.relu(tf.nn.conv2d(conv1_1, W['conv1_2'], strides=s1, padding='SAME') + B['conv1_2'])
	pool1 = tf.nn.max_pool(conv1_2, ksize=s2, strides=s2, padding='SAME')
	
	# sytle layer: conv2_1
	conv2_1 = tf.nn.relu(tf.nn.conv2d(pool1,   W['conv2_1'], strides=s1, padding='SAME')+ B['conv2_1'])
	conv2_2 = tf.nn.relu(tf.nn.conv2d(conv2_1, W['conv2_2'], strides=s1, padding='SAME')+ B['conv2_2'])
	pool2 = tf.nn.max_pool(conv2_2, ksize=s2, strides=s2, padding='SAME')

	# sytle layer: conv3_1
	conv3_1 = tf.nn.relu(tf.nn.conv2d(pool2,   W['conv3_1'], strides=s1, padding='SAME')+ B['conv3_1'])
	conv3_2 = tf.nn.relu(tf.nn.conv2d(conv3_1, W['conv3_2'], strides=s1, padding='SAME')+ B['conv3_2'])
	conv3_3 = tf.nn.relu(tf.nn.conv2d(conv3_2, W['conv3_3'], strides=s1, padding='SAME')+ B['conv3_3'])
	conv3_4 = tf.nn.relu(tf.nn.conv2d(conv3_3, W['conv3_4'], strides=s1, padding='SAME')+ B['conv3_4'])
	pool3 = tf.nn.max_pool(conv3_4, ksize=s2, strides=s2, padding='SAME')

	# sytle layer: conv4_1
	conv4_1 = tf.nn.relu(tf.nn.conv2d(pool3,   W['conv4_1'], strides=s1, padding='SAME')+ B['conv4_1'])
	# content layer: conv4_2
	conv4_2 = tf.nn.relu(tf.nn.conv2d(conv4_1, W['conv4_2'], strides=s1, padding='SAME')+ B['conv4_2'])
	conv4_3 = tf.nn.relu(tf.nn.conv2d(conv4_2, W['conv4_3'], strides=s1, padding='SAME')+ B['conv4_3'])
	conv4_4 = tf.nn.relu(tf.nn.conv2d(conv4_3, W['conv4_4'], strides=s1, padding='SAME')+ B['conv4_4'])
	pool4 = tf.nn.max_pool(conv4_4, ksize=s2, strides=s2, padding='SAME')

	# sytle layer: conv5_1
	conv5_1 = tf.nn.relu(tf.nn.conv2d(pool4,   W['conv5_1'], strides=s1, padding='SAME')+ B['conv5_1'])
	conv5_2 = tf.nn.relu(tf.nn.conv2d(conv5_1, W['conv5_2'], strides=s1, padding='SAME')+ B['conv5_2'])
	conv5_3 = tf.nn.relu(tf.nn.conv2d(conv5_2, W['conv5_3'], strides=s1, padding='SAME')+ B['conv5_3'])
	conv5_4 = tf.nn.relu(tf.nn.conv2d(conv5_3, W['conv5_4'], strides=s1, padding='SAME')+ B['conv5_4'])
	
	return conv4_2, [conv1_1, conv2_1, conv3_1, conv4_1, conv5_1]

def preprocess(imgData, meanColor):
	# In VGG model, the creators subtracted the average of each of the (R,G,B) channels
	# image data size: height  x width x chanel
	# but return size: 1 x height  x width x chanel
	return np.array([imgData - meanColor])

# compute style features in feedforward mode
def getcontentFeature(imgData):	
	imgData = preprocess(imgData, meanColor)
	g = tf.Graph()
	with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
		# placeholder for a image
		imgInput = tf.placeholder('float', shape=imgData.shape)
		
		# create deep learning model and put image's placeholder into
		# and then get conv4_2 layer (content layer)
		conv4_2, _ = createModel(imgInput)
		
		# Run tensorflow: feed image data into the model
		# and then get output from conv4_2 layer
		feauture = conv4_2.eval(feed_dict={imgInput: imgData})		
		return feauture

def getAllStyleFeatures(styleData):
	styleData = preprocess(styleData, meanColor)
	g = tf.Graph()
	with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
		# placeholder for a image
		imgInput = tf.placeholder('float', shape=styleData.shape)
		
		# create deep learning model and put image's placeholder into it
		# and then get style layer: [conv1_1, conv2_1, conv3_1, conv4_1 ,conv5_1]		
		_, styleLayer = createModel(imgInput)
		
		# Run tensorflow: feed style data into the model 
		# and then get output from layers: [conv1_1, conv2_1, conv3_1, conv4_1 ,conv5_1]
		output = sess.run(styleLayer, {imgInput: styleData})

		# Compute style feature map
		styleFeatures={}		
		for index, features in enumerate(output):						
			chanel = features.shape[3]
			features = np.reshape(features, (-1, chanel))
			# Gram matrix
			gram = np.matmul(features.T, features) / features.size
			styleFeatures[index] = gram
			assert styleFeatures[index].shape == (chanel, chanel)					
		
		return styleFeatures

from operator import mul
def _tensor_size(tensor):    
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def getStyleBlendWeights(styleBlendWeights=None ):	
    #if styleBlendWeights is None:
        # default is equal weights
        #styleBlendWeights = 1.0/len(styleDataList)        
    return 1

def trainModel(imgData, contentFeature, allStyleFeautures, allStep):			
	with tf.Graph().as_default():		
		imgShape = (1,) + imgData.shape
		initImg = tf.random_normal(imgShape) * 0.256
		# size: 1 x height x weight x weight
		#initImg = preprocess(initImg, meanColor )
		#initImg = initImg.astype('float32')		

		# variable will to be uppadated values while train the model each step
		image = tf.Variable(initImg)
		
		# create deep learning model and put image's variable into it
		# and then get layer: conv4_2, [conv1_1, conv2_1, conv3_1, conv4_1, conv5_1]
		conv4_2, styleLayer= createModel(image) 					
		
		# compute content loss
		contentLoss = CONTENT_WEIGHT * (2 * tf.nn.l2_loss(conv4_2 - contentFeature) / contentFeature.size)
		
		# compute style loss
		styleLoss = 0		
		styleLosses = []
		for index, layer in enumerate(styleLayer):				
			_, height, width, chanel = map(lambda i: i.value, layer.get_shape())     
			feats = tf.reshape(styleLayer[index], (-1, chanel))										
			layerSize = height * width * chanel
			gram = tf.matmul(tf.transpose(feats), feats) / layerSize
			styleGram = allStyleFeautures[index]
			styleLosses.append(2 * tf.nn.l2_loss(gram - styleGram) / styleGram.size)
		
		styleBlendWeights = getStyleBlendWeights();
		styleLoss += STYLE_WEIGHT * styleBlendWeights * reduce(tf.add, styleLosses)
		
		# total variation denoising
		tv_y_size = _tensor_size(image[:,1:,:,:])
		tv_x_size = _tensor_size(image[:,:,1:,:])
		tvLoss = TV_WEIGHT * 2 * (
				(tf.nn.l2_loss(image[:,1:,:,:] - image[:,:imgShape[1]-1,:,:]) /
				    tv_y_size) +
				(tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:imgShape[2]-1,:]) /
				    tv_x_size))
		
		# all loss function
		loss = contentLoss + styleLoss + tvLoss

		# optimizer setup
		# TensorFlow doesn't support L-BFGS (the original authors used),
		# so use Adam for optimizer
		optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
		
		sess = tf.Session()
		init = tf.global_variables_initializer()
		sess.run(init)
		
		bestLoss = float('inf')
		bestImg = None		
		
		for step in range(0,allStep):
			_, resultImg, currentLoss = sess.run([optimizer, image, loss])
			if  currentLoss < bestLoss:
				bestLoss = currentLoss
				bestImg = resultImg					
			
			resultImg = restoreImage(bestImg)
			
			if step %2 == 0:
				print('step %d | loss %.2f' % (step, currentLoss))
			if step %10 == 0:
				#saver = tf.train.Saver()	# save your model					
				#saver.save(sess, DIR_IMAGE_STEP, global_step=step)
				#saver.save(sess, DIR_IMAGE_STEP)
				scipy.misc.imsave("output.jpg", resultImg)
				
		return resultImg

# saver.restore(sess, ckpt.model_checkpoint_path)               

def imread3d(path): 
	# read image from file name 
	# and return array in 3 dimensions height  x width x chanel
	img = scipy.misc.imread(path).astype(np.float)
	if len(img.shape) == 2: # gray scale
		img = np.dstack((img,img,img))		
	return img 

def restoreImage(imgData):
	img = imgData[0] 		# because shape is 1 x height  x width x chanel
	img = img + meanColor 	# plus color mean(R,G,B)
	img = np.clip(img, 0, 255).astype(np.uint8)
	return img
	
def resizeImgData(imgData):	
	height, width, _ = imgData.shape
	# get shorter edge
	shortEdge = min([height , width]) 	
	# crop a image to square image: height  = width	
	marginY = int((height  - shortEdge) / 2)
	marginX = int((width - shortEdge) / 2)	
	cropImg = imgData[marginY: marginY + shortEdge, marginX: marginX + shortEdge]	
	# resize to 224, 224
	resizedImg = scipy.misc.imresize(cropImg, (224, 224))
	return resizedImg

def createImg():
	start_time = time.time() # start timmer
	#fileName = "1-content.jpg"
	fileName = "pic_4.jpg"
	sytleName = "the_scream.jpg"
	imgData = imread3d(fileName)
	imgData = resizeImgData(imgData)		# to: 244 x 244 x chanel
	assert imgData.shape[0:2] == (224, 224)

	styleData = imread3d(sytleName)
	styleData = resizeImgData(styleData)	# to: 244 x 244 x chanel
	assert styleData.shape[0:2] == (224, 224)

	# get content feature map
	contentFeature = getcontentFeature(imgData) 
	# get all style features map
	allStyleFeautures = getAllStyleFeatures(styleData)

	#initImg = np.random.randn( *imgData.shape)  *  0.256
	# waiting for many hours
	resultImg = trainModel(imgData, contentFeature, allStyleFeautures,1000)
	print("Creating image inished: %ds" % (time.time() - start_time))

	scipy.misc.imsave("output.jpg", resultImg)
	plt.imshow(resultImg)
	plt.show()

##  This code use 1 style, But in original code can use more than 1 styles
if __name__ == '__main__':
	createImg()
