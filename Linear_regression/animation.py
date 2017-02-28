from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def visualize(datasetX, datasetY, fxList):
	#fig, ax1 = plt.subplots()
	fig = plt.figure()
	
	#plt.gcf().canvas.set_window_title("Linear Regression")
	#fig.set_facecolor('#FFFFFF')
	#ax1 = fig.add_subplot(1,2,1)
	#ax2 = fig.add_subplot(1,2,2)

	#lnPlot, lnFx, lnFinal = ax1.plot([], [], 'bo', [], [], 'g-', [], [], 'r-',animated=True)
	lnPlot, lnFx, lnFinal = plt.plot([], [], 'bo', [], [], 'g-', [], [], 'r-',animated=True)
	totalFx = np.shape(fxList)[0]	# all fx	
		
	def init():
		#ax1.set_title('Linear Regression')
		#ax1.set_xlim(min(datasetX), max(datasetX))
		#ax1.set_ylim(min(datasetY), max(datasetY))
		#plt.title('Linear Regression')
		plt.title('Linear Regression')
		plt.xlim(min(datasetX), max(datasetX))
		plt.ylim(min(datasetY), max(datasetY))
		return lnPlot, lnFx, lnFinal

	def update(step):
		lnPlot.set_data(datasetX, datasetY)		
		fx = fxList[step]			
		lnFx.set_data(datasetX,fx)		
		if step == totalFx-1:
			print("plot graph finish")
			lnFinal.set_data(datasetX, fx)
			
		return lnPlot, lnFx, lnFinal
	
	step = range(0, totalFx)
	print("Ploting graph\nwaiting.........")
	ani = FuncAnimation(fig, update, 
						init_func=init, frames=step, repeat=True, blit=True)	
	plt.show()
