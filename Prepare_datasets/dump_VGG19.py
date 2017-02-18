import numpy as np
import scipy.io
import pickle

 
VGG_PATH = "D:\MyProject\machine-learning\Convolutional_neural_network\imagenet-vgg-verydeep-19"
# Use VGG-Network: https://arxiv.org/abs/1409.1556 (paper)
# http://www.vlfeat.org/matconvnet/pretrained/ (.mat)
dataVGG = scipy.io.loadmat(VGG_PATH)	
dataLayer = dataVGG['layers'][0]
assert dataLayer.shape == (43,) # all layer

# get color mean
mean = dataVGG['normalization'][0][0][0]
assert mean.shape == (224, 224, 3)
	
# the average color: Red, Green, Blue should be [123.68, 116.779, 103.939])
meanColor = np.mean(mean, axis=(0, 1))

W={}; B={}
for index, layer in enumerate(dataLayer):
	dd = layer[0][0]	
	if len(dd) <= 2: 
		# show 'relu layer' names
		print(dd[1]) 
	elif dd[3] == 'pool':		
		# show 'max pool' layer names
		print(dd[3]) 		
	else:		
		name = dd[3][0]
		weights, bias = dd[0][0]		
		# show names of conventional layer and fully connected layer
		print(name , " | weights:", weights.shape, " | bias:",bias.shape) 		
		#if not name.startswith('conv') :  # but select save weights and bias of conventional layer only
		#	continue		
		W[name] = weights
		B[name] = bias				
		
#print all layer (43) in VGG 19 model
"""
layer name		Size's weights			Size's bias 
['conv1_1'] 	(3, 3, 3, 64) 			(1, 64)
['relu1_1']
['conv1_2'] 	(3, 3, 64, 64) 			(1, 64)
['relu1_2']
['pool']
['conv2_1'] 	(3, 3, 64, 128) 		(1, 128)
['relu2_1']
['conv2_2'] 	(3, 3, 128, 128) 		(1, 128)
['relu2_2']
['pool']
['conv3_1'] 	(3, 3, 128, 256) 		(1, 256)
['relu3_1']
['conv3_2'] 	(3, 3, 256, 256) 		(1, 256)
['relu3_2']
['conv3_3'] 	(3, 3, 256, 256) 		(1, 256)
['relu3_3']
['conv3_4'] 	(3, 3, 256, 256) 		(1, 256)
['relu3_4']
['pool']
['conv4_1'] 	(3, 3, 256, 512) 		(1, 512)
['relu4_1']
['conv4_2'] 	(3, 3, 512, 512) 		(1, 512)
['relu4_2']
['conv4_3'] 	(3, 3, 512, 512) 		(1, 512)
['relu4_3']
['conv4_4'] 	(3, 3, 512, 512) 		(1, 512)
['relu4_4']
['pool']
['conv5_1'] 	(3, 3, 512, 512) 		(1, 512)
['relu5_1']
['conv5_2'] 	(3, 3, 512, 512) 		(1, 512)
['relu5_2']
['conv5_3'] 	(3, 3, 512, 512) 		(1, 512)
['relu5_3']
['conv5_4'] 	(3, 3, 512, 512) 		(1, 512)
['relu5_4']
['pool']
['fc6'] 		(7, 7, 512, 4096) 		(1, 4096)
['relu6']
['fc7'] 		(1, 1, 4096, 4096) 		(1, 4096)
['relu7']
['fc8'] 		(1, 1, 4096, 1000) 		(1, 1000)
['prob']
"""

# dump all weights and bias	
def dumpData():
	pickle.dump( W, open( "weights.p", "wb" ) )
	pickle.dump( B, open( "bias.p", "wb" ) )

	WW = pickle.load( open( "weights.p", "rb" ) )
	BB = pickle.load( open( "bias.p", "rb" ) )

	# Testing
	print("\nAfter read from pickle file")
	print("\nShape of weights")
	for key, value in sorted(WW.items()):
		print(key, value.shape)

	print("\nShape of bias")
	for key, value in sorted(BB.items()):
		print(key, value.shape)

if __name__ == '__main__':	
	pass
	#dumpData()