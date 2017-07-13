import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def show_examplePic(img, label, title , label_range=range(0,10)):	
	fig, axarr = plt.subplots(2, 5)
	axList = np.reshape(axarr, (2*5,))
	plt.gcf().canvas.set_window_title(title)
		
	for ax_index, num in enumerate(label_range):	
		ax = axList[ax_index]						
		ax.set_title("Label: %d" % num)
		index_list = np.where(label == num)[0]
		selected_imgList = img[index_list]
		random_index = np.random.randint(0, selected_imgList.shape[0])
		ax.set_axis_off()
		ax.imshow(selected_imgList[random_index])			
	plt.tight_layout()
	plt.show()

# Copy codes from: https://keras.io/datasets/ (14/06/2017)

##### CIFAR10 small image classification ##### 
# Dataset of 50,000 32x32 color training images, labeled over 10 categories, and 10,000 test images.
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
"""
Returns:
2 tuples:
x_train, x_test: uint8 array of RGB image data with shape (num_samples, 3, 32, 32).
y_train, y_test: uint8 array of category labels (integers in range 0-9) with shape (num_samples,).
"""
print("++++++ Dataset: CIFAR10 small image ++++++")
print("Shape (x_train, y_train) : ", x_train.shape, y_train.shape)
print("Shape (x_test, y_test) : ", x_test.shape, y_test.shape)
show_examplePic(x_train, y_train, title="Dataset: CIFAR10 small image")
print()


##### CIFAR100 small image classification ##### 
# Dataset of 50,000 32x32 color training images, labeled over 100 categories, and 10,000 test images.
from keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
"""
Returns:
2 tuples:
x_train, x_test: uint8 array of RGB image data with shape (num_samples, 3, 32, 32).
y_train, y_test: uint8 array of category labels with shape (num_samples,).

Arguments:
label_mode: "fine" or "coarse".
"""
print("++++++ Dataset: CIFAR100 small image ++++++")
print("Shape (x_train, y_train) : ", x_train.shape, y_train.shape)
print("Shape (x_test, y_test) : ", x_test.shape, y_test.shape)
random_label = np.random.randint(0, 100,size=10)
show_examplePic(x_train, y_train, title="Dataset: CIFAR100 small image", label_range=random_label)
print()


##### IMDB Movie reviews sentiment classification ##### 
# Dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative). Reviews have been preprocessed, and each review is encoded as a sequence of word indexes (integers). For convenience, words are indexed by overall frequency in the dataset, so that for instance the integer "3" encodes the 3rd most frequent word in the data. This allows for quick filtering operations such as: "only consider the top 10,000 most common words, but eliminate the top 20 most common words".
# As a convention, "0" does not stand for a specific word, but instead is used to encode any unknown word.
from keras.datasets import imdb
(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
"""
Returns:
2 tuples:
x_train, x_test: list of sequences, which are lists of indexes (integers). If the num_words argument was specific, the maximum possible index value is num_words-1. If the maxlen argument was specified, the largest possible sequence length is maxlen.
y_train, y_test: list of integer labels (1 or 0).

Arguments:
path: if you do not have the data locally (at '~/.keras/datasets/' + path), it will be downloaded to this location.
num_words: integer or None. Top most frequent words to consider. Any less frequent word will appear as oov_char value in the sequence data.
skip_top: integer. Top most frequent words to ignore (they will appear as oov_char value in the sequence data).
maxlen: int. Maximum sequence length. Any longer sequence will be truncated.
seed: int. Seed for reproducible data shuffling.
start_char: int. The start of a sequence will be marked with this character. Set to 1 because 0 is usually the padding character.
oov_char: int. words that were cut out because of the num_words or skip_top limit will be replaced with this character.
index_from: int. Index actual words with this index and higher.
"""
print("++++++ Dataset: IMDB Movie reviews sentiment ++++++")
print("Shape (x_train, y_train) : ", x_train.shape, y_train.shape)
print("Shape (x_test, y_test) : ", x_test.shape, y_test.shape)
print()


##### Reuters newswire topics classification ##### 
# Dataset of 11,228 newswires from Reuters, labeled over 46 topics. As with the IMDB dataset, each wire is encoded as a sequence of word indexes (same conventions).
from keras.datasets import reuters
(x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz",
                                                         num_words=None,
                                                         skip_top=0,
                                                         maxlen=None,
                                                         test_split=0.2,
                                                         seed=113,
                                                         start_char=1,
                                                         oov_char=2,
                                                         index_from=3)
"""
The specifications are the same as that of the IMDB dataset, with the addition of:

test_split: float. Fraction of the dataset to be used as test data.

This dataset also makes available the word index used for encoding the sequences:
word_index = reuters.get_word_index(path="reuters_word_index.json")

Returns: 
A dictionary where key are words (str) and values are indexes (integer). eg. word_index["giraffe"] might return 1234.

Arguments:
path: if you do not have the index file locally (at '~/.keras/datasets/' + path), it will be downloaded to this location.
"""
print("++++++ Dataset: Reuters newswire topics ++++++")
print("Shape (x_train, y_train) : ", x_train.shape, y_train.shape)
print("Shape (x_test, y_test) : ", x_test.shape, y_test.shape)
print()


##### MNIST database of handwritten digits ##### 
# Dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
"""
Returns:
2 tuples:
x_train, x_test: uint8 array of grayscale image data with shape (num_samples, 28, 28).
y_train, y_test: uint8 array of digit labels (integers in range 0-9) with shape (num_samples,).

Arguments:
path: if you do not have the index file locally (at '~/.keras/datasets/' + path), it will be downloaded to this location.
"""
print("++++++ Dataset: MNIST database of handwritten digits ++++++")
print("Shape (x_train, y_train) : ", x_train.shape, y_train.shape)
print("Shape (x_test, y_test) : ", x_test.shape, y_test.shape)
show_examplePic(x_train, y_train, title="Dataset: MNIST database of handwritten digits")
print()


##### Boston housing price regression dataset ##### 
# Dataset taken from the StatLib library which is maintained at Carnegie Mellon University.
# Samples contain 13 attributes of houses at different locations around the Boston suburbs in the late 1970s. Targets are the median values of the houses at a location (in k$).
from keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
"""
Arguments:
path: path where to cache the dataset locally (relative to ~/.keras/datasets).
seed: Random seed for shuffling the data before computing the test split.
test_split: fraction of the data to reserve as test set.
Returns: Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test).
"""
print("++++++ Dataset: Boston housing price dataset ++++++")
print("Shape (x_train, y_train) : ", x_train.shape, y_train.shape)
print("Shape (x_test, y_test) : ", x_test.shape, y_test.shape)
print()
column_head = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE" , "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
df =pd.DataFrame(columns=column_head, data=x_train)
df["MEDV"] = y_train
print(df.head())
