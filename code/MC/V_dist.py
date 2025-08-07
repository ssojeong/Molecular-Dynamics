import matplotlib.pyplot as plt
import torch
import sys
import numpy as np
import glob
import seaborn as sns
from scipy.stats import gaussian_kde

def mcsteps(u,start_step, npar,rho,temp, ylabel, mean=None, std=None):
    '''plot of potential energy at different temp or combined temp for nsamples ( train/ valid ) '''
    fig, axes = plt.subplots(1, u.shape[0], figsize=(16, 4))
    fig.suptitle(r'mc steps at npar={} $\rho$={} T={}'.format(npar,rho,temp), fontsize=15)
    for i, ax in enumerate(axes.flatten()):
        ax.set_title(r'mean U {:.3f}, std U {:.3f}'.format(mean[i],std[i]))
        ax.plot(u[i]/npar, 'k-', label='s{}'.format(i))
        ax.set_xlabel('mcs', fontsize=20)

        # ax.set_xticks(np.arange(0,u[i].shape[0]+1,50000),
        #               labels=np.arange(start_step,u[i].shape[0]+start_step+1 ,50000),fontsize=8)
        if mean is not None:
            ax.axhline(y=torch.mean(u[i]/npar), color='red', ls='--')
        if i % u.shape[0] == 0:
            ax.set_ylabel(ylabel, fontsize=15)
        ax.legend()
    plt.tight_layout()
    plt.show()


def dist(e, npar,rho, saved_dir):
    xmin=torch.min(e)
    xmax=torch.max(e)

    temp_list = [0.01, 0.1,0.2,0.4,0.6]
    fig, axes = plt.subplots(1, e.shape[0], figsize=(16, 4))
    fig.suptitle(r'Energy distribution at npar={} $\rho$={} (20 samples save 1000 mc steps)'.format(npar, rho), fontsize=15)
    for i, ax in enumerate(axes.flatten()):
        print(e[i].reshape(-1).shape)
        ax.hist(e[i].reshape(-1),bins=100,density=True, edgecolor='black', label=f'T={temp_list[i]}')  # displot

        kde = gaussian_kde(e[i].reshape(-1))
        x_vals =  np.linspace(torch.min(e[i]), torch.max(e[i]), 200)
        ax.plot(x_vals, kde(x_vals), color='darkblue', linewidth=2) #label='KDE (smoothed curve)'

        ax.set_xlabel(r'Potential', fontsize=15)
        # plt.xlim([xmin-0.1, xmax+0.1])
        ax.legend()

    plt.tight_layout()
    plt.show()
    # plt.savefig(saved_dir + '_n{}rho{}T{}_dist.png'.format(npar,rho,temp), bbox_inches='tight', dpi=200)
    # plt.close()

if __name__ == '__main__':
    #python V_dist.py ../../data_sets/gen_by_MC 2 0.03
    #
    torch.manual_seed(9745452)

    argv = sys.argv
    filename = argv[1]
    npar = int(argv[2])
    rho = argv[3]

    U_append = []
    accRatio_append = []
    spec_append = []
    file_path = sorted(glob.glob(filename + f'/n{npar}rho{rho}T*_U_ACCRatio_spec.pt'))

    for infile in file_path:
        U, accRatio, spec = torch.load(infile)
        U_sum = torch.sum(U)
        U2_sum = torch.sum(U * U)
        # cal_spec = (U2_sum / Nsum - U_sum * U_sum / Nsum / Nsum) / temp / temp / npar
        # assert(torch.mean(cal_spec-spec)< 1e-6), f'not match cal spec and saved spec...{torch.mean(cal_spec-spec):3e}'
        U_append.append(U)

    U_append = torch.stack(U_append) # shape [5, nsamples,nsteps]

    # perm_indx = torch.randperm(U_append.shape[1])
    # idx = perm_indx[:,4]

    # for s in idx:
    dist(U_append,  npar,rho, '')
