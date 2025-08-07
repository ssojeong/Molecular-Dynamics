import torch
import numpy as np
from utils.pbc import pbc
from utils.mydevice import mydevice

class isocahedron:

    # we don't use angle for 3D. in 2D hexgrid we use angle offset
    def __init__(self, b_list, a_list=None):

        assert len(b_list)==1,'only use one layer of grid'
        b = b_list[0]

        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        # Define the 12 vertices of a regular icosahedron
        vertices = np.array([
                            [0, 1, phi],
                            [0, -1, phi],
                            [1, phi, 0],
                            [-1, phi, 0],
                            [phi, 0, 1],
                            [-phi, 0, 1]
                           ])
        self.dim = 3 # 20250807

        # Add the negative counterparts to complete the 12 vertices
        vertices = np.r_[vertices, -vertices]

        # Normalize the vertices to lie on a unit sphere
        # (optional, but common for icospheres)
        vertices = vertices*b / np.linalg.norm(vertices, axis=1, keepdims=True)
        normalized_vertices = torch.from_numpy(vertices)

        self.all_grids = mydevice.load(normalized_vertices)
        self.ngrids = 12

    def __call__(self,q,l_list):
        '''make_grids function to shift 6 grids points at (0,0) to each particle position as center'''

        l_list = torch.unsqueeze(l_list, dim=2)
        # l_list.shape is [nsamples, nparticles, 1, DIM=3]

        l_list = l_list.repeat_interleave(self.all_grids.shape[0], dim=2)
        # l_list.shape is [nsamples, nparticles, ngrids, DIM=3]

        q_list = torch.unsqueeze(q, dim=2)
        # q_list.shape is [nsamples, nparticles, 1, DIM=(x coord, y coord, z coord)]

        grids_ncenter = self.all_grids + q_list  # broadcast
        # all_grids.shape = [12,3] = [ngrids,3]
        # grids_ncenter.shape is [ngrids, 3] + [nsamples, nparticles, 1, DIM=3] => [nsamples, nparticles, ngrids, DIM=3]

        #self.show_grids_nparticles(q, grids_ncenter, l_list[0,0,0], 'before')
        pbc(grids_ncenter, l_list)  # pbc - for grids
        #self.show_grids_nparticles(q, grids_ncenter,l_list[0,0,0], 'before')

        grids_ncenter = grids_ncenter.view(-1, q.shape[1] * self.all_grids.shape[0], q.shape[2])
        # shape is [nsamples, nparticles * ngrids, DIM=(x,y,z)]
        return grids_ncenter


