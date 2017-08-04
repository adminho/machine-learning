from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def visualize(datasetX, datasetY, fxList , accuracyList, lossList, title='Regression problems'):
	#fig, ax1 = plt.subplots()
	fig = plt.figure()
	
	plt.gcf().canvas.set_window_title('Regression problems')
	#fig.set_facecolor('#FFFFFF')
	#ax1 = fig.add_subplot(2,1,1)
	#ax2 = fig.add_subplot(2,2,3)
	#ax3 = fig.add_subplot(2,2,4)
	ax1 = fig.add_subplot(1,2,1)
	ax2 = fig.add_subplot(2,2,2)
	ax3 = fig.add_subplot(2,2,4)
		
	linePlot, lineFx, lineFinal = ax1.plot([], [], 'bo', [], [], 'g-', [], [], 'r-',animated=True)
	lineAccuracy, = ax2.plot([], [], 'b-',label="training accuracy")        
	lineLoss, = ax3.plot([], [], 'g-',label="training loss")
	totalFx = np.shape(fxList)[0]	# all fx	
	
	def init():
		ax1.set_title(title)		
		#margin = int( (min(datasetX) - max(datasetX))/10 )
		ax1.set_xlim(min(datasetX) , max(datasetX) )
		#margin = int( (min(datasetY) - max(datasetY))/10 )
		ax1.set_ylim(min(datasetY) , max(datasetY) )				
		
		ax2.set_title("Accuracy")
		ax2.set_xlim(0, len(accuracyList)) # initial value only, autoscaled after that
		ax2.set_ylim(0, max(accuracyList) +5) 			# not autoscaled
		
		ax3.set_title("Loss")
		ax3.set_xlim(0, len(lossList)) # initial value only, autoscaled after that
		ax3.set_ylim(0, max(lossList) + 0.1) # not autoscaled		
		plt.tight_layout()
		return linePlot, lineFx, lineFinal, lineAccuracy, lineLoss

	def update(step):
		#plt.tight_layout()
		# for ax1
		linePlot.set_data(datasetX, datasetY)	
		fx = fxList[step]		
		lineFx.set_data(datasetX, fx)		
		# for ax2 
		lineAccuracy.set_data(range(0, step), accuracyList[0: step])
		# for ax3
		lineLoss.set_data(range(0, step), lossList[0: step])
		
		if step == totalFx-1: # display last line
			print("plot graph finish")			
			lineFinal.set_data(datasetX, fx)							
		elif step == 0 :
			linePlot.set_data([], [])				
			lineFx.set_data([], [])
			lineFinal.set_data([], [])
			lineAccuracy.set_data([], [])
			lineLoss.set_data([], [])
			return init()
				
		return linePlot, lineFx, lineFinal, lineAccuracy, lineLoss
	
	step = range(0, totalFx)
	print("Ploting graph\nwaiting.........")
	ani = FuncAnimation(fig, update, 
						init_func=init, frames=step, repeat=True, blit=True)	
	plt.show()
