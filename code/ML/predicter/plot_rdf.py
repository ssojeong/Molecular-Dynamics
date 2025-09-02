import sys 
sys.path.append( '../../')

import torch

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    hist_list = ['../../../../Data/LLUF/300k_gromacs_histo.pt',
                 '../../../../Data/LLUF/300k_LLUF_10k_histo.pt',
                 '../../../../Data/LLUF/gromacs_rdf.pt']
    name = ['Gromacs', 'LUFNet', 'Gromacs Control']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
    for f, name in zip(hist_list, name):
        data = torch.load(f, weights_only=False)
        # print(data.keys())
        ax1.plot(data['edge_centers'], data['gr'], label=name)
        try:

            ax2.plot(data['edge_centers'], data['counts'] / data['counts'].sum(), label=name)
        except KeyError:
            pass

    plt.suptitle('100k steps in Gromacs / 10 k steps in LLUF')
    ax1.legend()
    ax2.legend()
    ax1.grid()
    ax2.grid()
    plt.show()

