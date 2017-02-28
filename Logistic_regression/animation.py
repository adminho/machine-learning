from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def visualize(X1A, X2A, X1B, X2B, X1_all, X2_all, fxList):
	fig, ax1 = plt.subplots()
	#fig = plt.figure()
	plt.gcf().canvas.set_window_title("Drawing")
	fig.set_facecolor('#FFFFFF')
	#ax1 = fig.add_subplot(1,2,1)
	plotA, plotB, lnFx, lnFinal = ax1.plot([], [], 'ro', [], [], 'bo', [], [], 'g-',[], [], 'k-', animated=True)
	totalFx = np.shape(fxList)[0]	# all fx	
	
	def init():
		ax1.set_title('Logistic regression')
		ax1.set_xlim(min(X1_all)*1, max(X1_all)*1)
		ax1.set_ylim(min(X2_all)*1, max(X2_all)*1)
		return plotA, plotB, lnFx, lnFinal 

	def update(step):	
		plotA.set_data(X1A, X2A)
		plotB.set_data(X1B, X2B)
		
		fx = fxList[step]				
		lnFx.set_data(X1_all, fx)		
		if step == totalFx-1:
			print("plot graph finish")
			lnFinal.set_data(X1_all, fx)
					
		return plotA, plotB, lnFx, lnFinal 
	
	step = range(0, totalFx)
	print("Ploting graph\nwaiting.........")
	ani = FuncAnimation(fig, update, 
						init_func=init, frames=step, repeat=True, blit=True)	
	plt.show()
