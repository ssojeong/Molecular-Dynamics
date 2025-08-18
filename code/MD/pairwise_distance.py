import sys
# sys.path.append( '../')

#import matplotlib.pyplot as plt
from utils.pbc import pairwise_dq_pbc
from utils.get_paired_distance_indices import get_paired_distance_indices
import glob
import torch
import numpy as np
from scipy.stats import gaussian_kde

def pair_dq(q_list,l_list):
    '''
    # pair_dq(q_list):

        use to compute the pair distance between particles

        :param q_list: with only time point , shape = [batch,nparticles,dim=2 or 3]
        :return: square pair distance
    '''

    nsamples, nparticle, dim = q_list.shape

    pair_dq = pairwise_dq_pbc(q_list,l_list) # shape = [nsamples,nparticles,nparticles,dim=2 or 3]

    idx = get_paired_distance_indices.get_indices(pair_dq.shape)
    pair_dq = get_paired_distance_indices.reduce(pair_dq, idx)
    pair_dq = pair_dq.view([nsamples, nparticle, nparticle - 1, dim])
    # dd shape = [batch,nparticles,nparticles-1]
    dd = torch.sqrt(torch.sum(pair_dq * pair_dq, dim=-1))

    return dd

def plot_pairs(dd,npar,rho, boxsize):

    # plot compare square pair-wise distance
    temp_list = [0.9]
    print(boxsize.item())
    print(dd.reshape(-1).shape)
    plt.title(r"boxsize={:.3f} npar={} $\rho$={}".format(boxsize.item(), npar, rho), fontsize=16)
    # plt.title('dq min {:.2f}; max {:.2f}'.format(torch.min(dd[i])*boxsize.item(),torch.max(dd[i])*boxsize.item()))

    plt.hist(dd.reshape(-1).cpu().numpy(),bins=100,density=True, edgecolor='black', label=f'T={temp_list[0]}')
    kde = gaussian_kde(dd.reshape(-1).cpu().numpy())
    x_vals =  np.linspace(torch.min(dd.reshape(-1).cpu()), torch.max(dd.reshape(-1).cpu()), 200)
    plt.plot(x_vals, kde(x_vals), color='darkblue', linewidth=2) #label='KDE (smoothed curve)'

    plt.xlabel(r'$r$',fontsize=16)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # python pairwise_distance.py ../../data_sets/gen_by_MC 2 0.03

    argv = sys.argv
    filename = argv[1]
    npar = int(argv[2])
    rho = argv[3]

    print((npar/float(rho))**(1./3))

    file_path = sorted(glob.glob(filename + f'/s20n{npar}rho{rho}T*.pt'))

    dd_list = []
    for infile in file_path:
        data = torch.load(infile)
        qpl_list = data['qpl_trajectory'] # shape [nsamples, (q,p,l), 1, npar, dim]
        print(qpl_list.shape)

        q = qpl_list[:, 1, 0, :, :].clone().detach()
        boxsize = qpl_list[:, 2, 0, :, :].clone().detach()

        dd_list.append(pair_dq(q,boxsize))

    dd_list = torch.stack(dd_list)

    plot_pairs(dd_list,npar,rho,torch.mean(boxsize))

