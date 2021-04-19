import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import sys
import math

# epoch:0 train_loss:1 valid_loss:2 train_dq/boxsz:3 valid_dq/boxsz:4
# train_dq:5 valid_dq:6 train_dp:7 valid_dp:8 time:9

if __name__ == '__main__':

    argv = sys.argv

    if len(argv) != 4:
        print('usage <programe> <y1> <y2> <title>')
        quit()

    y1   = int(argv[1])
    y2   = int(argv[2])
    name = argv[3]

    # paramters
    nsamples = "480,000/96,000"
    nparticle = 4
    long_tau = 0.1
    short_tau = 0.001
    rho = 0.1
    lr = 0.01
    NN = "5->128->128->16->2"
    optim = 'SGD'
    activation = 'tanh'
    time = '250'
    titan = '02'

    boxsize = math.sqrt(nparticle/rho)
    data = np.genfromtxt('n{}st{}lt{}_nsampled_loss.txt'.format(nparticle,short_tau, long_tau))

    x = data[:,0]
    loss = data[:,y1]
    val_loss = data[:,y2]


    fig2 =plt.figure()
    ax2 = fig2.add_subplot(111)
    #ax2.ticklabel_format(style='sci', scilimits=(0,0), axis='y')
    ax2.plot(x,loss,'blue',label='train',zorder=2)
    ax2.plot(x,val_loss,'orange',label='valid',zorder=2)
    ax2.set_xlabel('epoch',fontsize=30)
    ax2.set_ylabel('Loss',fontsize=30)
    ax2.tick_params(labelsize=20)
    #ax2.set_ylim([1.0986,1.0988])
    ax2.legend(loc='upper right',fontsize=20)
    plt.title(name,fontsize=20)
    anchored_text = AnchoredText('nsamples={} nparticle={} boxsize={:.3f} \nlarge time step = {}, short time step {} \nNN input 5  output 2 opt {} lr {} activation tanh() \nNN {} time per epoch= {}s titan {}'.format(nsamples,nparticle, boxsize,long_tau,short_tau,optim,lr,NN,time,titan), loc= 'upper left', prop=dict(fontweight="normal", size=12))
    ax2.add_artist(anchored_text)
    plt.grid()
    plt.show()
    plt.close()