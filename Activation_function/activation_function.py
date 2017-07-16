import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1/(1+np.exp(-x))

def tanh(x):
	#return 2/(1+np.exp(-2x)) - 1
	return 2 * sigmoid(2*x) - 1

def relu(x):
	return x *(x>=0) 
	
def leaky_relu(X, leak=0.2):
	f1 = 0.5 * (1 + leak)
	f2 = 0.5 * (1 - leak)
	return f1 * X + f2 * np.abs(X)

x = np.arange(-50,50,0.1)	
y1 = sigmoid(x)
y2 = tanh(x)
y3 = relu(x)
y4 = leaky_relu(x)

# create 4 graph
fig = plt.figure()
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)

ax1.title.set_text('Input')
ax2.title.set_text('Sigmoid function')
ax3.title.set_text('TanH function')
ax4.title.set_text('ReLU function')
ax5.title.set_text('Leaky ReLU function')

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
diplay_xy_axis(ax5)

# show all graphs
ax1.plot(x, x,'k-')
ax2.plot(x, y1,'r-')
ax3.plot(x, y2,'g-')
ax4.plot(x, y3,'b-')
ax5.plot(x, y4,'c-')
plt.tight_layout()
plt.show()