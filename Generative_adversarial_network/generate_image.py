# paper: https://arxiv.org/abs/1406.2661
# reference code example from : https://github.com/osh/KerasGAN/blob/master/MNIST_CNN_GAN.ipynb
# reference blog: https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/

from __future__ import print_function
import os.path

import numpy as np
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K

import matplotlib.pyplot as plt

generative_h5_file = "mnist_generator.h5"
discriminative_h5_file = "mnist_discriminator.h5"

def load_weights(model, h5_file):
	try:
		if os.path.exists(h5_file):
			print("\nLoaded model(weights) from file: %s" % (self.h5_file))
			model.load_weights(self.h5_file)	
	except Exception as inst:
		print(inst)
	return model
			
# Build Discriminative model 
def build_discriminative_model(shape):
	model = Sequential()
	model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding="same", activation='relu', input_shape=shape))
	model.add(LeakyReLU(0.2))
	model.add(Dropout(0.25))
	model.add(Conv2D(filters=512, kernel_size=(5, 5), strides=(2, 2), padding="same", activation='relu'))
	model.add(LeakyReLU(0.2))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(units=256))
	model.add(LeakyReLU(0.2))
	model.add(Dropout(0.25))
	model.add(Dense(units=1, activation='sigmoid'))
	# Output is binary classification
	
	model = load_weights(model, discriminative_h5_file)			
	#opt = Adam(1e-5)
	opt = Adam(1e-4)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# If is_train==False, it will be freeze weights in the discriminator (don't update weights)
def enable_train(discriminator, is_train=True):
	discriminator.trainable = is_train
	for layer in discriminator.layers:
		layer.trainable = is_train

#build Generative model ...
def build_generative_model(discriminator, shape=(100, )):
	generator = Sequential()
	generator.add(Dense(input_shape=shape, units=200 * 14 * 14))
	generator.add(BatchNormalization())
	generator.add(Activation('relu'))
	
	if K.image_dim_ordering() == 'th': 
		# backend is Theano
		# Image dimension = chanel x row x column
		generator.add(Reshape( (200, 14, 14) ))
	else: 
		# 'tf' backend is Tensorflow
		# Image dimension = row x column x chanel
		generator.add(Reshape( (14, 14, 200) ))
		
	generator.add(UpSampling2D(size=(2, 2)))
	generator.add(Conv2D(filters=100, kernel_size=(3, 3), padding="same", kernel_initializer='glorot_uniform'))
	generator.add(BatchNormalization())
	generator.add(Activation('relu'))
	generator.add(Conv2D(filters=50, kernel_size=(3, 3), padding="same", kernel_initializer='glorot_uniform'))
	generator.add(BatchNormalization())	
	generator.add(Activation('relu'))
	# images color (gray) are between 0 to 1 (depends on sigmoid values)
	generator.add(Conv2D(filters=1, kernel_size=(1, 1), padding="same", kernel_initializer='glorot_uniform', activation='sigmoid'))
	
	generator = load_weights(generator, generative_h5_file)
	#++++++Finish build generative_model mode +++++++++++++++++++++++

	#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	# build stack of and discriminative_model and generative_model
	# don't train discriminator
	enable_train(discriminator,False)	
	input = generator.inputs[0]		
	output = discriminator(generator(input))
	# this for training generative_model only
	train_generator = Model(input, output)
	
	#opt = Adam(1e-6)
	opt = Adam(1e-3)
	train_generator.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return generator, train_generator
	
def save_accuracy(acc_dis, acc_gen):
	plt.figure(figsize=(10,8))	
	plt.plot(acc_dis, label='discriminitive accuracy')
	plt.plot(acc_gen, label='generative accuracy')
	plt.legend()	
	plt.savefig("MNIST_ACCURACY.png")

def save_genImage(generator, n_ex=16,dim=(4,4), figsize=(10,10) ):
	noise = np.random.uniform(0,1,size=[n_ex,100])
	generated_images = generator.predict(noise)	
	plt.figure(figsize=figsize)
	for i in range(generated_images.shape[0]):
		plt.subplot(dim[0],dim[1],i+1)
		img = []
		if K.image_dim_ordering() == 'th': 
			# backend is Theano
			# Image dimension = chanel x row x column 
			img = generated_images[i,0,:,:]
		else:
			# 'tf' backend is Tensorflow
			# Image dimension = row x column x chanel
			img = generated_images[i,:,:,0] # tensorflow
		plt.imshow(img)
		plt.axis('off')
	plt.tight_layout()	
	plt.gray()
	print("Picture color: min = %f and max = %f" % (np.min(generated_images), np.max(generated_images)))
	plt.savefig("MNIST_GENERATE.png")

# save accuracy and generated image to graph
acc_dis = []
acc_gen = []

def train_GAN(X_train, discriminator, generator, nb_epoch=5000, num_sampling=32):
	for e in range(nb_epoch): 
		print("\n===================== Iterator : %s===================" % e)
		if e == 0: num_sampling = 300 # frist training		
		# create noise 
		Z_noise = np.random.uniform(0,1,size=[num_sampling, 100])	

		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		#+++++++++++++++++++++ Training Discriminative Model +++++++++++++++++++++++++
		# Random minibath from real image 
		X_real = X_train[np.random.randint(0, X_train.shape[0], size=num_sampling),:,:,:]
		assert X_real.shape[0] == num_sampling
		assert X_real.shape[1:] == X_train.shape[1:]
	
		# Make generated images (fake images) from input noise	
		X_fake = generator.predict(Z_noise)
		# concatenate fake and real images
		X_concat = np.concatenate((X_real, X_fake))		
		assert X_concat.shape[0] == X_real.shape[0] + X_fake.shape[0]

		# Create target datasets (label 0 or 1)
		Y_real = np.ones(num_sampling) 			# label real image to 1
		Y_fake = np.zeros(num_sampling) 			# label fake image to 0
		Y_label = np.append(Y_real, Y_fake) 	# combine all labels to a vector
		assert Y_label.shape[0] == X_concat.shape[0]
		assert Y_label.shape[0] == Y_real.shape[0] + Y_fake.shape[0]
		
		# Unfreeze weights of discriminator model before training discriminator
		enable_train(discriminator)	
		# Train discriminator on generated images (fake image) and real images		
		if e == 0:
			# frist training
			discriminator.fit(X_concat, Y_label, verbose =0, epochs=1, batch_size=128)			
				
		scores = discriminator.train_on_batch(X_concat, Y_label )
		acc_dis.append(scores[1])			
		print("Discriminator: loss = %f and accuracy = %f" % ( scores[0], scores[1]))		
		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		
		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		#+++++++++++++++++++++ Training Generative Model +++++++++++++++++++++++++
		# don't train discriminator (freeze weights of discriminative model)
		enable_train(discriminator,False)
		# make all fake image to label 1 (fake label)
		Y_fake = np.ones(num_sampling)
		scores = None		
		for i in range(0,1):
			# train Generator-Discriminator stack on input noise		
			scores = train_generator.train_on_batch(Z_noise, Y_fake)
		acc_gen.append(scores[1])
		print("Generator: loss = %f and accuracy = %f" % ( scores[0], scores[1]))		
		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
				
		if e%2 == 0:
			# save all data to graphs
			save_accuracy(acc_dis, acc_gen)
			save_genImage(generator)
			# Backup model
			discriminator.save(discriminative_h5_file)
			generator.save(generative_h5_file)			

def reshapeCNNInput(X): 
	exampleNum, W, W = X.shape
	# change shape of image data
	if K.image_dim_ordering() == 'th': 
		# backend is Theano
		# Image dimension = chanel x row x column (chanel = 1, if it is RGB: chanel = 3)
		XImg = X.reshape(exampleNum, 1, W, W)
	else: 
		# 'tf' backend is Tensorflow
		# Image dimension = row x column x chanel (chanel = 1, if it is RGB: chanel = 3)
		XImg = X.reshape(exampleNum, W, W, 1)		
	return XImg
	
def prepare_Dataset():
	#X_test, Y_train, Y_test => unused
	(X_train, Y_train), (X_test, Y_test) = mnist.load_data() 
	# use mini examples for training and testing
	X_train = X_train[0:500]	
	X_train = reshapeCNNInput(X_train)
	print('X_train shape:', X_train.shape)
	print('X_test shape:', X_test.shape)
		
	# Normalized
	X_train = X_train.astype('float32')
	X_train /= 255	
	print ('Min and max train dataset: %s , %s' % (np.min(X_train), np.max(X_train) ) )
	return X_train

if __name__ == "__main__":
	# Prepare dataset
	print("Looad and prepare datasets.....")
	X_train = prepare_Dataset()

	print("Building .....")
	# input shape to discriminative_model is th:(chanel, row, column) or tf:(row, column, chanel)
	discriminator = build_discriminative_model(X_train.shape[1:]) 
	generator, train_generator = build_generative_model(discriminator)
	
	print("Training....")
	train_GAN(X_train, discriminator, generator, nb_epoch=1000, num_sampling=32)

	#++++++++++++++++++++++++++++show model summary++++++++++++++++++++++++++++++
	print(generator.summary())
	print(train_generator.summary())
	print(discriminator.summary())

##Generative model
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1 (Dense)              (None, 3200)              323200
_________________________________________________________________
batch_normalization_1 (Batch (None, 3200)              12800
_________________________________________________________________
activation_1 (Activation)    (None, 3200)              0
_________________________________________________________________
reshape_1 (Reshape)          (None, 4, 4, 200)         0
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 8, 8, 200)         0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 8, 8, 100)         180100
_________________________________________________________________
batch_normalization_2 (Batch (None, 8, 8, 100)         400
_________________________________________________________________
activation_2 (Activation)    (None, 8, 8, 100)         0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 8, 50)          45050
_________________________________________________________________
batch_normalization_3 (Batch (None, 8, 8, 50)          200
_________________________________________________________________
activation_3 (Activation)    (None, 8, 8, 50)          0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 8, 1)           51
=================================================================
Total params: 561,801
Trainable params: 555,101
Non-trainable params: 6,700
_________________________________________________________________
"""

# GAN model
# Build Generator-Discriminator stack 
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_1_input (InputLayer)   (None, 100)               0
_________________________________________________________________
sequential_1 (Sequential)    (None, 8, 8, 1)           561801
_________________________________________________________________
sequential_2 (Sequential)    (None, 1)                 3808769
=================================================================
Total params: 4,370,570
Trainable params: 4,363,870
Non-trainable params: 6,700
_________________________________________________________________
"""

#Discriminative model 
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_4 (Conv2D)            (None, 4, 4, 256)         6656
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 4, 4, 256)         0
_________________________________________________________________
dropout_1 (Dropout)          (None, 4, 4, 256)         0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 2, 2, 512)         3277312
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 2, 2, 512)         0
_________________________________________________________________
dropout_2 (Dropout)          (None, 2, 2, 512)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 2048)              0
_________________________________________________________________
dense_2 (Dense)              (None, 256)               524544
_________________________________________________________________
leaky_re_lu_3 (LeakyReLU)    (None, 256)               0
_________________________________________________________________
dropout_3 (Dropout)          (None, 256)               0
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 257
=================================================================
Total params: 3,808,769
Trainable params: 3,808,769
Non-trainable params: 0
_________________________________________________________________
"""