# My idea from: 
# http://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html
# http://scikit-learn.org/stable/auto_examples/svm/plot_iris.html#sphx-glr-auto-examples-svm-plot-iris-py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors as mcolors
import matplotlib.mlab as mlab
import os.path
import scipy.misc

import time
import datetime

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.recurrent import SimpleRNN, LSTM, GRU 
from keras.models import Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D
import keras

# my modules
from history import TrainingHistory

X_train = [[-0.4326, 1.1909], 
	[3.0, 4.0],
	[0.1253 , -0.0376   ],
	[0.2877 ,   0.3273  ],
	[-1.1465 ,   0.1746 ],
	[1.8133 ,   1.0139  ],
	[2.7258 ,   1.0668  ],
	[1.4117 ,   0.5593  ],
	[4.1832 ,   0.3044  ],
	[1.8636 ,   0.1677  ],
	[0.5 ,   3.2  ],
	[0.8 ,   3.2  ],
	[1.0 ,   -2.2  ],
	[2.1 ,   -4.2  ],
	[3.5 ,   -4.7  ],
	[3.0 ,   -5.0  ]]

Y_train = [ 1 , 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test, Y_test = X_train, Y_train

def build_MLP(features):
	model = Sequential()		
	# L2 is weight regularization penalty, also known as weight decay, or Ridge
	model.add(Dense(input_dim=features, units=6) )
	model.add(Activation("tanh"))	
	model.add(Dense(units=1)) 	
	# now model.output_shape == (None, 10)
	# note: `None` is the batch dimension.	
	#
	model.add(Activation("sigmoid"))
		
	# algorithim to optimize the models (train model)
	# compute loss with function: binary crossentropy
	#opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	opt = keras.optimizers. Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(optimizer=opt,
			  loss='binary_crossentropy',
			  metrics=['accuracy'])
	return model

model = build_MLP(X_train.shape[1])
his = TrainingHistory()
def training_model(model, step_visual=0, visual=None):
	model.fit(X_train, Y_train, epochs=10, verbose=0, callbacks=[his])	
	visual.update_line(his.loss, his.accuracy)
	
	iterator = ((step_visual+1)*10)
	if iterator%50 == 0:
		print("============= Iterator %d ================" %  iterator)
		# evaluate after trained
		scores = model.evaluate(X_train, Y_train, verbose=0)
		print("Evalute model: %s = %.4f" % (model.metrics_names[0] ,scores[0]))
		print("Evalute model: %s = %.4f" % (model.metrics_names[1] ,scores[1]*100))
	return model
		
class Visualization():	
	def __init__(self, model, X_train, Label_train, title, dpi=70):
		fig = plt.figure(figsize=(19.20,10.80), dpi=dpi)		
		plt.gcf().canvas.set_window_title(title)
		fig.set_facecolor('#FFFFFF')
		ax1 = fig.add_subplot(121)
		ax2 = fig.add_subplot(222)		
		ax3 = fig.add_subplot(224)
		ax1.grid(False) # toggle grid off
		ax2.grid(False) # toggle grid off
		ax3.grid(False) # toggle grid off
		
		ax1.set_title("Classify 2 groups")
		ax2.set_title("Loss")		
		ax3.set_title("Accuracy")
				
		self.ax1, self.ax2, self.ax3 = ax1, ax2, ax3		
		self.fig =fig
		
		self.model = model		
		self.xy_label = []		
		self.loss = []
		self.accuracy = []
				
		# There are 2 classess
		class_label = range(0, max(Label_train)+1)
		self.scatter_list = [ self.ax1.scatter([], []) for i in range(0,len(class_label))]
				
		# seperate 2 class of X_train
		for number in class_label:
			index_label = np.where(Label_train == number)[0]
			xy = X_train[index_label]		
			self.xy_label.append(xy)
			
		h = .3  # step size in the mesh
		# create a mesh to plot in
		x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
		y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
		xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
				
		self.xx = xx
		self.yy = yy
		
		self.line2, = ax2.plot([], [])		
		self.line3, = ax3.plot([], [])		
				
		ax1.set_xticks(())
		ax1.set_yticks(())
		ax1.set_xlabel('X values')
		ax1.set_ylabel('Y values')
		ax1.set_xlim(xx.min(), xx.max())
		ax1.set_ylim(yy.min(), yy.max())
		plt.tight_layout()
		
	def init(self):		
		self.ax2.set_ylim(0, 1)   # not autoscaled 	
		self.ax3.set_ylim(0, 110)  # not autoscaled			
		return self.scatter_list, self.line2, self.line3
			
	def update(self):
		m = ['v', 's']
		colors = ['navy', 'orangered' ]
		
		# Put the result into a color plot
		Z = model.predict(np.c_[self.xx.ravel(), self.yy.ravel()])				
		Z = Z.reshape(self.xx.shape)
		con1 = self.ax1.contourf(self.xx, self.yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)	
			
		for index, data in enumerate(self.xy_label):		
			self.scatter_list[index] = self.ax1.scatter(data[:, 0], data[:, 1], c=colors[index] , marker=m[index])
			#self.scatter_list[index] = self.ax1.scatter(data[:, 0], data[:, 1], cmap=plt.cm.coolwarm , marker=m[index])
		
		# rescale x		
		x2 = range(0, len(self.loss))		
		x3 = range(0, len(self.accuracy))				
		self.ax2.set_xlim(0, max(x2)+1)
		self.ax3.set_xlim(0, max(x3)+1)
		# plot loss and accuracy
		self.line2.set_data(x2, self.loss)
		self.line3.set_data(x3, self.accuracy)	
		return self.scatter_list, con1, self.line2, self.line3
	
	def update_line(self, loss, accuracy):
		self.loss = loss
		self.accuracy = accuracy
		
	def train(self, training_model, iterations, save_movie=False):	
		def animate_func(step):	
			if step >= iterations-1:				 				
				print("\n+++++++++++++++++ Finish training +++++++++++++++++")
				plt.close()			
			else:								
				self.model = training_model(self.model, step_visual=step, visual=self)						
				plt.pause(0.501) # makes the UI a little more responsive 
			return self.update()
	
		ani = animation.FuncAnimation(self.fig, animate_func, iterations, 
						init_func=self.init, repeat=False, interval=16, blit=False)
		if save_movie:
			# save all picture to mp4 file
			mywriter = animation.FFMpegWriter(fps=24, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-profile:v', 'high', '-tune', 'animation', '-crf', '18'])		
			ani.save("example_2_class.mp4", writer=mywriter)
		else:
			plt.show()

visual = Visualization(model, X_train, Y_train, title="Example: binary classification")
visual.train(training_model, iterations=70, save_movie=False)