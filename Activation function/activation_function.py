import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1/(1+np.exp(-x))

def tanh(x):
	#return 2/(1+np.exp(-2x)) - 1
	return 2 * sigmoid(2*x) - 1

def relu(x):
    return x *(x>=0) 
    
x = np.arange(-50,50,0.1)	
y1 = sigmoid(x)
y2 = tanh(x)
y3 = relu(x)

# create 4 graph
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.title.set_text('Input')
ax2.title.set_text('Sigmoid function')
ax3.title.set_text('TanH function')
ax4.title.set_text('ReLU function')

# set scale on y axis
ax2.set_ylim([-0.1, 1.1])
ax3.set_ylim([-1.1, 1.1])
#ax4.set_ylim([-0.2, x])

def diplay_xy_axis(ax):
    #ax.set_aspect('equal')
    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')

diplay_xy_axis(ax1)
diplay_xy_axis(ax2)
diplay_xy_axis(ax3)
diplay_xy_axis(ax4)

# show all graphs
ax1.plot(x, x,'k-')
ax2.plot(x, y1,'r-')
ax3.plot(x, y2,'g-')
ax4.plot(x, y3,'b-')
plt.show()