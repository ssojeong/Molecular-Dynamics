import torch

class dpair_pbc:

    _obj_count = 0

    def __init__(self):

        dpair_pbc._obj_count += 1
        assert (dpair_pbc._obj_count <= 3),type(self).__name__ + " has more than one object"


    def paired_grid_q(self, phase_space, grids_list, boxsize):
        '''
        function to measure distance btw particle and grid and use this to calculate dphi_fields and v_grids

        Parameters
        ----------
        q_state : shape is [nsamples, nparticle, DIM=(x,y)]
        grids_list : torch.tensor
                shape is [nsamples, nparticle* ngrids, DIM=(x,y)]
        Returns
        ----------
        paired_grid_q : torch.tensor
                 paired_grid_q.shape is [nsamples, nparticle, nparticle * ngrids, DIM]
        dd : torch.tensor
                dd.shape is [nsamples,nparicle, nparticle * ngrids]
        '''

        q_state = phase_space.get_q()

        q_state = torch.unsqueeze(q_state,dim = 2)
        # shape is [nsamples, nparticle, 1, DIM=(x,y)]
        grids_list = torch.unsqueeze(grids_list, dim=1)
        # shape is [nsamples, 1, nparticle* ngrids, DIM=(x,y)]
        paired_grid_q =  grids_list - q_state
        # paired_grid_q.shape is [nsamples, nparticle, nparticle * ngrids, DIM]

        phase_space.adjust_real_q(paired_grid_q, boxsize)

        dd = torch.sqrt(torch.sum(paired_grid_q * paired_grid_q, dim=-1))
        # dd.shape is [nsamples, nparticle, nparticle * ngrids]

        return paired_grid_q, dd


