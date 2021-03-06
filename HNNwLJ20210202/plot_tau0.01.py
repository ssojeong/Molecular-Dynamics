import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('nsamples50_tau0.01_loss.txt')

x = range(0,58700)
loss = data[:,1]
loss = loss[:58700]

fig2 =plt.figure()
ax2 = fig2.add_subplot(111)
#ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
ax2.plot(x,loss,'blue',label='train',zorder=2)
ax2.set_xlabel('epoch',fontsize=30)
ax2.set_ylabel('Loss',fontsize=30)
ax2.tick_params(labelsize=20)
#ax2.set_ylim([1.0986,1.0988])
ax2.legend(loc='upper right',fontsize=20)
plt.title(r'large time step 0.01 / short time step 0.01',fontsize=30)
plt.grid()
plt.show()
plt.close()
