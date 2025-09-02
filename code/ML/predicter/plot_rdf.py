import sys 
sys.path.append( '../../')

import torch

import matplotlib.pyplot as plt
import numpy as np

# sys.modules['numpy._core'] = np.core

if __name__ == '__main__':
    print(torch.__version__)
    print(np.__version__)
    hist_list = ['../../../../Data/LLUF/300k_gromacs_histo.pt',
                 '../../../../Data/LLUF/300k_LLUF_10k_histo.pt',
                 '../../../../Data/LLUF/gromacs_rdf.pt']
    name = ['Gromacs ', 'LUFNet', 'Gromacs Control']
    for f, name in zip(hist_list, name):
        data = torch.load(f, weights_only=False, map_location='cpu')
        gr = data['gr']
        print(f, gr.shape, data['edge_centers'].shape)
        plt.plot(data['edge_centers'], gr, label=name)

    plt.title('100k steps in Gromacs / 10 k steps in LLUF')
    plt.legend()
    plt.grid()
    plt.show()

