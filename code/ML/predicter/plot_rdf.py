import sys 
sys.path.append( '../../')

import torch

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    hist_list = ['../../../../Data/LLUF/300k_gromacs_histo.pt', '../../../../Data/LLUF/300k_LUFNet_10k_histo.pt']
    name = ['Gromacs', 'LUFNet']
    for f, name in zip(hist_list, name):
        data = torch.load(f, weights_only=False)
        counts = data['counts']
        counts = counts / counts.sum()
        plt.plot(data['edge_centers'], counts, label=name)

    plt.title('100k steps in Gromacs / 10 k steps in LLUF')
    plt.legend()
    plt.grid()
    plt.show()

