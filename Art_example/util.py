import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def getAllImageData(path):
	img = scipy.misc.imread(path).astype(np.float)
	if len(img.shape) == 2: # grayscale		
		img = np.dstack((img,img,img))	
	return img
	
def getColorDataInPixel(imageData):			
	imageData = np.array(imageData)
	high, width, _ = imageData.shape
	
	# reshape size of image data to size: [high*width] x 3
	# 1 row per color information (RGB) in a pixel	
	image_train = imageData.reshape(high*width, -1)

	_, chanel = image_train.shape	
	if chanel >= 4: # for .png file (png format is RGBA, jpg format is RGB)						
		image_train = np.delete(image_train[:], 3, axis=1) # select RGB only
	
	return image_train/255 # normalized		

def getCoordTrain(high, width):		
	coordinates = np.zeros((high, width, 2))		
	for w in range(0,width): # get all coordinates (x,y)
		for h in range(0,high):			 
			coordinates[h][w][0] = (w-width/2)/width; # normalize x codinate			 
			coordinates[h][w][1] = (h-high/2)/high;   # normalize y codinate
	
	# reshape coordinates to size: [high*width] x 2
	coordinates  = np.array(coordinates)
	coord_train = coordinates.reshape(high*width , -1)
	return coord_train
	
def preShowImage(imageData):
	return np.clip(imageData, 0, 255).astype(np.uint8)
	
def restoreImage(colPredict, high, width):	
	# restore RGB values from normalized		
	colPredict = np.floor(255*colPredict)	
	# Restore shape of color_predicted to size: high x width x 3	
	imageData = colPredict.reshape((high, width , 3) )
	return imageData

import os
import os.path
import shutil

TEMP_PATH = "log_pic"
if os.path.exists(TEMP_PATH):
	shutil.rmtree(TEMP_PATH)	
os.makedirs(TEMP_PATH)

# Show the image that created	
def visualize(orgImage, savedPic, trainModel, coordTrain, colorTrain, maxStep, save_movie=False):
	fig = plt.figure()
	plt.gcf().canvas.set_window_title("Drawing")
	fig.set_facecolor('#FFFFFF')
	ax1 = fig.add_subplot(1,2,1)
	ax1.grid(False) # toggle grid off
	ax1.set_axis_off()
	ax1.set_title('Orginal Picture')

	ax2 = fig.add_subplot(1,2,2)
	ax2.grid(False) # toggle grid off
	ax2.set_axis_off()
	ax2.set_title('AI Painting')
	
	orgImage = preShowImage(orgImage)
	orgImax = ax1.imshow(orgImage, animated=True, cmap='binary', vmin=0.0, vmax=1.0, interpolation='nearest', aspect=1.0)
	showPicBegin = np.ones(orgImage.shape)
	paintImax = ax2.imshow(showPicBegin, animated=True, cmap='binary', vmin=0.0, vmax=1.0, interpolation='nearest', aspect=1.0)        
	
	# size of image data: high x width x RGB
	high, width, _ = orgImage.shape
	
	def initImg_func():		
		return orgImax, paintImax

	def showImage(colPredict):	
		imgData = restoreImage(colPredict, high, width)
		imgShow = preShowImage(imgData)
		paintImax.set_data(imgShow)	# draw a predicted image
		return imgShow

	def updateImg_func(step):		
		# Train your model (each step)		
		colPredict, correct, loss = trainModel(step, coordTrain, colorTrain)
		assert colPredict.shape == colorTrain.shape
		
		if correct > 99 or step == maxStep-1 :		
			print('Finised at step %d | loss %f  | accuracy %f' % (step, loss, correct))
			imgShow = showImage(colPredict) # visualize a image that AI is painting
			#ani.event_source.stop()	   # stop show image
			#saver.save(sess, saved_wieght_file, global_step=step)
			scipy.misc.imsave(savedPic, imgShow)
			return orgImax, paintImax
			
		if step %10 == 0:
			print('step %d | loss %f  | accuracy %f' % (step, loss, correct))
			showImage(colPredict)	# visualize a image that AI is painting	
			plt.savefig(os.path.join(TEMP_PATH, "step_%d" % (step+1) )) # save the figure
			
		return orgImax, paintImax
	
	ani = animation.FuncAnimation(fig, updateImg_func, frames=np.arange(0, maxStep),
                    init_func=initImg_func, repeat=False, blit=True)		
	if save_movie:
		# save all picture to mp4 file
		mywriter = animation.FFMpegWriter(fps=24, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-profile:v', 'high', '-tune', 'animation', '-crf', '18'])		
		ani.save("paint_example.mp4", writer=mywriter)
	else:
		plt.show()

