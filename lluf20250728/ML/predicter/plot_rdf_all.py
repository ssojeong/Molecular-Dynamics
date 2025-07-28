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
    gamma = float(argv[3])
    dpt = int(argv[4])
    region = argv[5]
    y1limf = float(argv[6])
    y1limb = float(argv[7])
    y2limf = float(argv[8])
    y2limb = float(argv[9])

    if (gamma == 0.0) or (gamma == 1.0) or (gamma == 10.0) or (gamma == 20.0):
        print('gamma {} float to int .... '.format(gamma))
        gamma = int(gamma)

    load_file = f'npar{npar}rho{rho}gamma{gamma}nstep10000_rdf.txt'

    with open(load_file) as f:
        rdf = np.genfromtxt(load_file, filling_values=np.nan)
    print('load rdf...',rdf.shape)
    print(rdf.shape)

    temp_list = [0.44, 0.46, 0.48, 0.5]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5, 5), gridspec_kw = {'wspace':0,'hspace':0.05})

    for j in np.arange(len(temp_list)):
       print('rdf mc', temp_list[j], '1st rmid ', rdf[j, 1], rdf[j, 2], 'gr', rdf[j,3], rdf[j,4],
                                    '2nd rmid', rdf[j, 5], rdf[j, 6], 'gr', rdf[j, 7], rdf[j, 8])
       print('rdf vv', temp_list[j], '1st rmid ', rdf[j, 9], rdf[j, 10], 'gr', rdf[j, 11], rdf[j, 12],
                                    '2nd rmif', rdf[j,13], rdf[j,14], 'gr', rdf[j, 15], rdf[j, 16])
       print('rdf ml', temp_list[j], '1st rmid ', rdf[j, 17],rdf[j, 18], 'gr', rdf[j, 19], rdf[j, 20],
                                    '2nd rmid', rdf[j,21], rdf[j,22], 'gr', rdf[j, 23], rdf[j, 24])

       if (npar == 64 or npar == 128) and region == 'lg' : #and j < len(temp_list)-1:
           print('plot n=128 models ....')
           print('ml128 ', temp_list[j],'1st rmid ', rdf[j, 25], rdf[j, 26], 'gr',rdf[j, 27], rdf[j, 28] ,
                                        '2nd rmid ', rdf[j, 29], rdf[j, 30], 'gr', rdf[j, 31], rdf[j, 32])

           axes[0].errorbar([j+1], rdf[j, 27], yerr=rdf[j, 28], capsize=5,elinewidth=0.5, color='r', markerfacecolor='none',marker='x',linestyle='none', markersize=12)
           axes[1].errorbar([j+1], rdf[j, 31], yerr=rdf[j, 32], capsize=5,elinewidth=0.5, color='r', markerfacecolor='none', marker='x', linestyle='none', markersize=12)
           # axes[0].errorbar([j+1], gr_n128_first, color='r', markerfacecolor='none',marker='x',linestyle='none', markersize=12)
           # axes[1].errorbar([j + 1], gr_n128_second, color='r', markerfacecolor='none', marker='x', linestyle='none', markersize=12)


       axes[0].errorbar([j+1], rdf[j, 3], yerr=rdf[j, 4], capsize=5,elinewidth=0.5,  color='k', markerfacecolor='none',marker='o',linestyle='none', markersize=12)
       axes[0].errorbar([j+1], rdf[j, 11], yerr=rdf[j, 12], capsize=5,elinewidth=0.5, color='k', markerfacecolor='none',marker='^',linestyle='none', markersize=12)
       axes[0].errorbar([j + 1], rdf[j, 19], yerr=rdf[j, 20], capsize=5, elinewidth=0.5, color='k',
                        markerfacecolor='none', marker='x', linestyle='none', markersize=12)

       axes[1].errorbar([j+1], rdf[j, 7], yerr=rdf[j, 8], capsize=5,elinewidth=0.5, color='k', markerfacecolor='none',marker='o',linestyle='none', markersize=12)
       axes[1].errorbar([j+1], rdf[j, 15], yerr=rdf[j, 16], capsize=5,elinewidth=0.5,  color='k', markerfacecolor='none',marker='^',linestyle='none', markersize=12)
       axes[1].errorbar([j + 1], rdf[j, 23], yerr=rdf[j, 24], capsize=5, elinewidth=0.5, color='k',
                        markerfacecolor='none', marker='x', linestyle='none', markersize=12)

       axes[0].set_ylim(y1limf,y1limb)
       # axes[0].set_yticks([y1limf, y1limb])

       axes[0].set_yticks([7, 10, 13])
       # axes[0].set_yticks([4.6, 5.8, 7])
       # axes[0].set_yticks([3.2, 3.6])

       axes[1].set_ylim(y2limf,y2limb)
       # axes[1].set_yticks([y2limf, y2limb])

       axes[1].set_yticks([1.5, 2, 2.5])
       # axes[1].set_yticks([1.6, 2.2, 2.8])
       # axes[1].set_yticks([1.5, 1.6])

       #ax.plot(np.array(nrmid), np.array(ngr), color='k',label='t={:.2f}'.format(input_seq_idx * tau_long + step[j] * tau_long))
       for ax in axes.flat:
        ax.grid(True, axis='x', linestyle='--')
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both',  right=False, labelleft=False)
        # ax.set_yticks([1, 1.5,  2])
        # ax.set_yticklabels([1,"",2])
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xticks(ticks=[1, 2, 3, 4], labels=temp_list,fontsize=14)

    # plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()

    # plt.show()
    saved_dir =  f'../../../data_sets/gen_by_ML/lt0.1dpt{dpt}_{region}/npar{npar}rho{rho}gamma{gamma}nstep10000_rdf.pdf'
    plt.savefig(saved_dir, bbox_inches='tight', dpi=200)
    plt.close()
