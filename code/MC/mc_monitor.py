import matplotlib.pyplot as plt
import torch
import sys
import numpy as np
import glob
import matplotlib.ticker as ticker

def split_mcsteps(u,npar,rho,temp, ylabel,start=0, mean=None, std=None):
    print(len(u))

    rows, cols = 5, 5 # 5, 5
    num_plots = 50   # 200
    plots_per_figure = 25  #25

    for fig_num in range((num_plots + plots_per_figure -1)//plots_per_figure):

        fig, axes = plt.subplots(rows, cols, figsize=(20,20))
        #fig.suptitle(r'every 100 mc steps at npar={} $\rho$={} T={}'.format(npar,rho,temp), fontsize=15)
        axes = axes.flatten()

        for i in range(plots_per_figure):

            global_index= fig_num * plots_per_figure+i
            print('global index', global_index)
            if global_index >= len(u):
                axes[i].axis('off')
                continue
            axes[i].plot(u[global_index][start:,0],u[global_index][start:,1]/npar, 'k-', label='s{}'.format(i))
            axes[i].set_xlabel('mcs', fontsize=15)

            axes[i].set_xlim(left= 0,right=20000)
            axes[i].set_ylim([-6, -4])
            #
            axes[i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

            if i % len(u)== 0:
               axes[i].set_ylabel(ylabel, fontsize=15)
            axes[i].legend()
    
        plt.tight_layout()
        plt.show()

def mcsteps(u,npar,rho,temp, ylabel, mean=None, std=None):
    '''plot of potential energy at different temp or combined temp for nsamples ( train/ valid ) '''

    plt.title(r'every 100 mc steps at npar={} $\rho$={} T={}'.format(npar,rho,temp), fontsize=15)
    plt.plot(u[0][:,0],u[0][:,1]/npar, 'k-', label='s{}'.format(0))
    plt.xlabel('mcs', fontsize=20)
    # ax.set_xlim(left=1000,right=None) 
    plt.ylabel(ylabel, fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.show()

# def mcsteps(u,npar,rho,temp, ylabel,start=0, mean=None, std=None):
#     '''plot of potential energy at different temp or combined temp for nsamples ( train/ valid ) '''
#     fig, axes = plt.subplots(2, len(u)//2, figsize=(16, 10))
#     fig.suptitle(r'every 100 mc steps at npar={} $\rho$={} T={}'.format(npar,rho,temp), fontsize=15)
#     for i, ax in enumerate(axes.flatten()):
#         print(i)
#         print(u[i].shape)
#         ax.plot(u[i][start:,0],u[i][start:,1]/npar, 'k-', label='s{}'.format(i))
#         ax.set_xlabel('mcs', fontsize=20)
#         # ax.set_xlim(left=1000,right=None)
#         ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
#
#         if mean is not None:
#             ax.axhline(y=torch.mean(u[i]/npar), color='red', ls='--')
#         if i % len(u)== 0:
#             ax.set_ylabel(ylabel, fontsize=15)
#         ax.legend()
#     plt.tight_layout()
#     plt.show()

if __name__ == '__main__':
    #python mc__monitor.py log/ 32 0.85 0.9
    #
    torch.manual_seed(9745452)

    argv = sys.argv
    filename = argv[1]
    npar = int(argv[2])
    rho = argv[3]
    temp = argv[4]
    start = 0

    e_append = []
    # file_path = sorted(glob.glob(filename + f'/n{npar}rho{rho}T*_U_ACCRatio_spec.pt'))
    file_path = sorted(glob.glob(filename + f'n512*.txt'))

    for infile in file_path:
        print(infile)
        e = np.genfromtxt(infile)

        if e.size == 0:
            print('Empty file, skipping')
            continue
        e_append.append(e)
        print(e.shape)

    # e_append = np.stack(e_append) # shape [nsamples,nsteps,2] ; 2 is time, energy
    #
    mcsteps(e_append, npar, rho, temp, '')

    # # for s in idx:
    # split_mcsteps(e_append,  npar,rho, temp, '', start)
