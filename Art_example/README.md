# Example: AI paints an image

This example shows the neural network (multilayer perceptrons) to paint an image

# Code Example

* [paint_tensorflow.py](paint_tensorflow.py) (implemented with Tensorflow)
* [paint_keras.py](paint_keras.py) (implemented with Keras)

# Motivation

My inspiration: http://cs.stanford.edu/people/karpathy/convnetjs/demo/image_regression.html


# Description


## My model structure 

* 7 hidden layers.
* Input layer have 2 ( x, y coordinate or pixels on an image).
* Output layer have 3 (red, breen and blue).

![AI architec](images/AI_Paint_architec.png)


## Concept

Input to the model are x, y coordinates (on an target image).

![AI architec](images/Input_Paint.png)

** In python, size of an image is height x width

      
If there is one input to the model.

![My_network_paint](images/My_network_paint1.png)

    
If there are many inputs (many examples) to the model.

![My_network_paint](images/My_network_paint2.png)

For target dataset to train my model, I use real an image as the target dataset. When neural network learns from the target image, it can predict 3 outputs are red, green, blue (RGB).


## Result

The picture on the left is original and the right hand side is created by my neural network.

