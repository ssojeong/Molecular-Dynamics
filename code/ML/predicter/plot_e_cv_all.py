import sys
sys.path.append( '../../')

import torch
from utils.system_logs              import system_logs
from utils.mydevice                 import mydevice
import math
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(34952)

    argv = sys.argv

    npar = int(argv[1])
    rho = argv[2]
    dim = int(argv[3])
    gamma = float(argv[4])
    dpt = int(argv[5])
    region = argv[6]
    y1limf = float(argv[7])
    y1limb = float(argv[8])
    y2limf = float(argv[9])
    y2limb = float(argv[10])

    if (gamma == 0.0) or (gamma == 1.0) or (gamma == 10.0) or (gamma == 20.0):
        print('gamma {} float to int .... '.format(gamma))
        gamma = int(gamma)

    load_file1 = f'../../analysis/{dim}d/npar{npar}rho{rho}gamma{gamma}_e.txt'
    load_file2 = f'../../analysis/{dim}d/npar{npar}rho{rho}gamma{gamma}_cv.txt'

    with open(load_file1) as f:
        e = np.genfromtxt(load_file1, filling_values=np.nan)
        # shape [temp_list, 5] ; 5 is [(idx, vv_mean, vv_std, ml_mean, ml_std)]
    print('load energy...',e.shape)
    print(e)
    with open(load_file2) as f:
        cv = np.genfromtxt(load_file2, filling_values=np.nan)
        # shape [temp_list, 5] ; 5 is [(idx, vv_mean, vv_std, ml_mean, ml_std)]
    print('load Cv...', cv.shape)
    print(cv)

    temp_list = [0.44, 0.46, 0.48, 0.5]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5, 5), gridspec_kw = {'wspace':0,'hspace':0.05})

    for j in np.arange(4): # 4 is len(temp_list)

       print('e vv', temp_list[j], e[j,1], e[j,2])
       # print('e ml', temp_list[j], e[j,3],e[j,4])
       print('cv vv', temp_list[j], cv[j, 1],cv[j, 2])
       # print('cv ml', temp_list[j], cv[j, 3],cv[j, 4])

       if (npar == 64 or npar == 128) and region == 'lg' :# and j < 3:
           print('')
           print('plot n=64 or 128 models ....')
           print('e ml', temp_list[j], e[j, 5], e[j, 6])
           print('cv ml', temp_list[j], cv[j, 5], cv[j, 6])
           print('')
           axes[0].errorbar([j+1], e[j, 5], yerr=e[j, 6], capsize=5,elinewidth=0.5, color='r', markerfacecolor='none',marker='x',linestyle='none', markersize=12)
           axes[1].errorbar([j+1], cv[j, 5], yerr=cv[j, 6], capsize=5,elinewidth=0.5, color='r', markerfacecolor='none', marker='x', linestyle='none', markersize=12)


       axes[0].errorbar([j+1], e[j, 1], yerr=e[j, 2], capsize=5,elinewidth=0.5,  color='k', markerfacecolor='none',marker='^',linestyle='none', markersize=12)
       axes[0].errorbar([j+1], e[j, 3], yerr=e[j, 4], capsize=5,elinewidth=0.5, color='k', markerfacecolor='none',marker='x',linestyle='none', markersize=12)

       axes[1].errorbar([j+1], cv[j, 1], yerr=cv[j, 2], capsize=5,elinewidth=0.5, color='k', markerfacecolor='none',marker='^',linestyle='none', markersize=12)
       axes[1].errorbar([j+1], cv[j, 3], yerr=cv[j, 4], capsize=5,elinewidth=0.5,  color='k', markerfacecolor='none',marker='x',linestyle='none', markersize=12)

       axes[0].set_ylim(y1limf,y1limb)
       axes[1].set_ylim(y2limf, y2limb)

       if region == 'g':
           print('phase g ... set y ticks ...')
           axes[0].set_yticks([0, 0.08]) # gas
           axes[1].set_yticks([0, 0.9, 1.8]) # gas

       if region == 'lg':
           print('phase lg ... set y ticks ...')
           axes[1].set_yticks([0, 1, 2, 3]) # liquid+gas

       for ax in axes.flat:
        ax.grid(True, axis='x', linestyle='--')
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both',  right=False, labelleft=False)
        # ax.set_yticks([1, 1.5,  2])
        # ax.set_yticklabels([1,"",2])
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xticks(ticks=[1, 2, 3, 4], labels=temp_list,fontsize=14)

    plt.tight_layout()

    saved_dir = f'../../analysis/{dim}d/figures/e_cv/npar{npar}rho{rho}gamma{gamma}nstep10000_e_cv.pdf'
    plt.savefig(saved_dir, bbox_inches='tight', dpi=200)
    plt.close()
