import pandas as pd
from os.path import join
import numpy as np

def prepare_dataset(csv_dataset,x_column_names, y_column_name, base_dir  = "" ):
	# read csv file with pandas module	
	df = pd.read_csv(join(base_dir, csv_dataset))
	
	print("First of 5 row in Dataset")
	print(df.head())	
	print("\nTail of 5 row in Dataset")
	print(df.tail())
	
	train_X = {}	
	for index, column in enumerate(x_column_names):		
		train_X[index] = df[column].reshape(-1,1)		# X (Input) training set
	
	train_Y = df[y_column_name].reshape(-1,1)			# Y (Output) training set	

	print("Size of training set Y: {}".format(train_Y.shape))
	return train_X, train_Y
