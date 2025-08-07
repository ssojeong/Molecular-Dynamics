import torch
from utils.mydevice import mydevice

class SingleGrid:

    def __init__(self):
        # grids_ncenter.shape is [6*nlayers, 2]
        grids_ncenter = torch.tensor([[0,0]]) # shape [1,2]
        self.all_grids = mydevice.load(grids_ncenter)
        self.ngrids = 1

    # ===================================================
    # for each particle, make the grid points around it
    # return all the grid points of all particles
    #
    def __call__(self,q,l_list):
        '''make_grids function to shift 1 grid point at (0,0) to each particle position as center'''

        l_list = torch.unsqueeze(l_list, dim=2)
        # l_list.shape is [nsamples, nparticles, 1, DIM]

        l_list = l_list.repeat_interleave(self.all_grids.shape[0], dim=2)

        q_list = torch.unsqueeze(q, dim=2)
        # q_list.shape is [nsamples, nparticles, 1, DIM=(x coord, y coord)]

        grids_ncenter = self.all_grids + q_list  # broadcast
        # all_grids.shape = [1,2]
        # grids_ncenter.shape is [1, 2] + [nsamples, nparticles, 1, DIM] => [nsamples, nparticles, 1, DIM=2]

        # dont need pbc: pbc(grids_ncenter, l_list)  # pbc - for grids
        # self.show_grids_nparticles(q, grids_ncenter,l_list[0,0,0])

        grids_ncenter = grids_ncenter.view(-1, q.shape[1] * self.all_grids.shape[0], q.shape[2])
        # shape is [nsamples, nparticles * ngrids, DIM=(x,y)]
        return grids_ncenter
    # ===================================================





