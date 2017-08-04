from keras.callbacks import Callback
import numpy as np

class TrainingHistory(Callback):
	def __init__(self):
		self.loss = []
		self.accuracy = []
		
	def on_train_begin(self, logs={}):
		pass
		
	def on_batch_end(self, batch, logs={}):		
		pass
			
	def on_epoch_end(self, batch, logs={}):
		self.loss = np.append(self.loss, logs.get('loss'))
		self.accuracy = np.append(self.accuracy, 100*logs.get('acc'))