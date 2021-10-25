import torch

class dpair_pbc:

    _obj_count = 0

    def __init__(self):

        dpair_pbc._obj_count += 1
        assert (dpair_pbc._obj_count == 1),type(self).__name__ + " has more than one object"


    def xi_shifts(self, xi_state):
        '''
        function to shift xi position as center 9 times that is no. of box

        Parameters
        ----------
        xi_state : torch.tensor
                dimensionless state
                shape is [nsamples, nparticle, DIM]

        Returns
        ----------
        xi_shift : torch.tensor
                shape is [nsampels, 9,nparticle,DIM]
        '''

        shifts = torch.tensor([[-1., -1.], [-1., 0.], [-1., 1.], [0., -1.], [0., 0.], [0., 1.], [1., -1.], [1., 0.], [1., 1.]])
        # shifts.shape = [9,DIM]

        xi_state = torch.unsqueeze(xi_state, dim=1)
        # xi_state.shape = [nsamples, 1, nparticle, DIM]

        shifts = torch.unsqueeze(shifts, dim=1)
        # shifts.shape = [9,1,DIM]

        return xi_state + shifts   # shape is [nsamples, 9, nparticle, DIM]

    def cdist(self, xi_state, grids_list):
        '''
        function to find shortest distance btw particle and grid among 9 boxes

        Parameters
        ----------
        xi_state : torch.tensor
                dimensionless state
                shape is [nsamples, nparticle, DIM]
        grids_list : torch.tensor
                dimensionless state
                shape is [nsamples, nparticle*grids18, DIM=(x,y)]

        Returns
        ----------
        dpairs_good : torch.tensor
                dpairs_good.shape is [nsamples,nparicle, npar*grids18]
        '''

        nsamples,nparticle, DIM = xi_state.shape
        _,ngrids18, DIM = grids_list.shape

        xi_pbc = self.xi_shifts(xi_state)
        # xi_pbc.shape is [nsamples,9,npar,DIM]

        xi_pbc = xi_pbc.reshape(nsamples,-1,DIM)
        # xi_pbc.shape is [nsamples,9*npar,DIM]

        # grids_list.shape is [nsamples, npar*grids18, DIM=(x,y)]
        dp = torch.cdist(xi_pbc, grids_list)
        # dp.shape is [nsamples, 9*npar, npar*grids18]

        dpairs = dp.reshape(nsamples, 9, nparticle, ngrids18)
        # dpairs.shape is [nsamples, 9, npar, npar*grids18]

        dpairs_good, _ = torch.min(dpairs, dim=1)
        # dpairs_good.shape is [nsamples, npar, npar*grids18]

        return dpairs_good


    def paired_grid_xi(self, xi_state, grids_list, boxsize=1.):
        '''
        method 2: function to measure distance btw particle and grid

        Parameters
        ----------
        xi_state : torch.tensor
                dimensionless state
                shape is [nsamples, nparticle, DIM]
        grids_list : torch.tensor
                dimensionless state
                shape is [nsamples, nparticle* ngrids, DIM=(x,y)]
        Returns
        ----------
        paired_grid_xi : torch.tensor
                dimensionless state
                 paired_grid_xi.shape is [nsamples,nparicle, grids]
        dd : torch.tensor
                dimensionless state
                dd.shape is [nsamples,nparicle, grids]
        '''

        xi_state = torch.unsqueeze(xi_state,dim = 2)
        grids_list = torch.unsqueeze(grids_list, dim=1)

        paired_grid_xi =  grids_list - xi_state
        # paired_grid_xi.shape is [nsamples, nparticle, nparticle * ngrids, DIM]

        indices = torch.where(torch.abs(paired_grid_xi)>0.5 * boxsize)
        paired_grid_xi[indices] = paired_grid_xi[indices] - torch.round(paired_grid_xi[indices])

        dd = torch.sqrt(torch.sum(paired_grid_xi * paired_grid_xi, dim=-1))
        # dd.shape is [nsamples, nparticle, nparticle * ngrids]

        return paired_grid_xi, dd

    def paired_grid_q(self, q_state, grids_list, boxsize):
        '''
        method 2: function to measure distance btw particle and grid

        Parameters
        ----------
        q_state : torch.tensor
                shape is [nsamples, nparticle, DIM]
        grids_list : torch.tensor
                shape is [nsamples, nparticle * ngrids, DIM=(x,y)]

        '''
        paired_grid_q, dd = self.paired_grid_xi( q_state, grids_list, boxsize)

        return paired_grid_q, dd

