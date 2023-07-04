from fields.dpair_pbc import dpair_pbc
import torch

class LJ_term:

    _obj_count = 0

    def __init__(self, epsilon, sigma):
        ''' LJ_term class for all potential and derivative of potential

        Parameters
        ----------
        epsilon : int
        sigma : int
        boxsize : float
        '''

        LJ_term._obj_count += 1
        assert (LJ_term._obj_count == 1),type(self).__name__ + " has more than one object"

        self._epsilon  = epsilon
        self._sigma    = sigma
        self.dpair_pbc = dpair_pbc()

        self._name = 'Lennard Jones Potential'
        print('LJ_term initialized : sigma ',sigma,' epsilon ',epsilon)

    def phi_fields(self, xi_space, grids_list): 
        '''
        function to get phi field each grid at particle position as center

        Parameters
        ----------
        xi_space : torch.tensor
                dimensionless state
                shape is [nsamples, nparticle, DIM]
        grids_list : torch.tensor
                dimensionless state
                shape is [nsamples, nparticle*ngrids, DIM=(x,y)]

        Returns
        ----------
        term : torch.tensor
                shape is [nsamples, nparticle*ngrids]
        '''

        xi_state = xi_space.get_q()
        # xi_state.shape is [nsamples, nparticle, DIM]
        boxsize = xi_space.get_boxsize()

        a12 = (4 * self._epsilon * pow(self._sigma, 12)) / pow(boxsize, 12)
        a6 = (4 * self._epsilon * pow(self._sigma, 6)) / pow(boxsize, 6)

        # # grids_list.shape is [nsamples, nparticle*ngrids, DIM=(x,y)]
        # d = self.dpair_pbc.cdist(xi_state, grids_list)
        # # d.shape is [nsamples, nparticle, nparticle * ngrids]

        _, d = self.dpair_pbc.paired_grid_xi(xi_state, grids_list)

        if __debug__:
            max_d = torch.sqrt(2 * torch.tensor(0.5) ** 2)  # this function dimensionless so that boxsize is 1
            assert ( (max_d > d ).all() == True), 'error pbc maximum distance ...'

        s12_ = 1 / pow(d, 12)
        s6_ = 1 / pow(d, 6)
        # s12.shape is [nsamples, nparticle, nparticle * ngrids]

        s12 = self.zero_phi_fields(s12_)
        s6 = self.zero_phi_fields(s6_)

        phi_fields = (a12 * torch.sum(s12, dim=1) - a6 * torch.sum(s6, dim=1))
        # phi_fields.shape is [nsamples, nparticle*ngrids]

        return phi_fields

    def zero_phi_fields(self, s12s6):

        nsamples, npar, npar_ngrids = s12s6.shape

        make_zero_phi = torch.reshape(s12s6, (nsamples, npar, npar, npar_ngrids//npar))
        # make_zero_phi.shape is [nsamples, nparticle, nparticle, ngrids]

        dy = torch.diagonal(make_zero_phi,0,1,2) # offset, nparticle, nparticle
        torch.fill_(dy,0.0)

        s12s6_reshape = torch.reshape(make_zero_phi, (nsamples, npar, npar_ngrids))

        return s12s6_reshape

    def derivate_phi_fields(self, xi_space, grids_list):
        '''
        function to get phi field each grid at particle position as center

        Parameters
        ----------
        xi_space : torch.tensor
                dimensionless state
                shape is [nsamples, nparticle, DIM]
        grids_list : torch.tensor
                dimensionless state
                shape is [nsamples, nparticle*ngrids, DIM=(x,y)]

        Returns
        ----------
        term : torch.tensor
                shape is [nsamples, nparticle*ngrids]
        '''

        xi_state = xi_space.get_q()
        # xi_state.shape is [nsamples, nparticle, DIM]
        boxsize = xi_space.get_boxsize()

        a12 = (4 * self._epsilon * pow(self._sigma, 12)) / pow(boxsize, 13)
        a6 = (4 * self._epsilon * pow(self._sigma, 6)) / pow(boxsize, 7)

        # # grids_list.shape is [nsamples, nparticle*ngrids, DIM=(x,y)]
        # d = self.dpair_pbc.cdist(xi_state, grids_list)
        # # d.shape is [nsamples, nparticle, nparticle * ngrids]

        delta_grid_xi, d = self.dpair_pbc.paired_grid_xi(xi_state, grids_list)
        # delta_grid_xi.shape is [nsamples, nparticle, nparticle * ngrids, DIM]
        # d.shape is [nsamples, nparticle, nparticle * ngrids]

        if __debug__:
            max_d = torch.sqrt(2 * torch.tensor(0.5) ** 2)  # this function dimensionless so that boxsize is 1
            assert ( (max_d > d ).all() == True), 'error pbc maximum distance ...'

        d = torch.unsqueeze(d,dim =-1)
        # d.shape is [nsamples, nparticle, nparticle * ngrids, 1]

        s12_ = -12 * (delta_grid_xi) / pow(d,14)
        s6_  = -6 * (delta_grid_xi) / pow(d,8)
        # s12.shape is [nsamples, nparticle, nparticle * ngrids, DIM]

        s12 = self.zero_derivate_phi_fields(s12_)
        s6 = self.zero_derivate_phi_fields(s6_)

        dphi_fields = a12 * torch.sum(s12, dim=1) - a6 * torch.sum(s6, dim=1)  # np.sum axis=1 j != k
        # dphidxi.shape is [nsamples, nparticle * ngrids, DIM=2]

        return dphi_fields

    def zero_derivate_phi_fields(self, s12s6):

        nsamples, npar, npar_ngrids, DIM = s12s6.shape

        make_zero_dphi = torch.reshape(s12s6, (nsamples, npar, npar, npar_ngrids//npar, DIM))
        # make_zero_phi.shape is [nsamples, nparticle, nparticle, ngrids, DIM]

        dy = torch.diagonal(make_zero_dphi,0,1,2) # offset, nparticle, nparticle
        torch.fill_(dy,0.0)

        s12s6_reshape = torch.reshape(make_zero_dphi, (nsamples, npar, npar_ngrids,DIM))

        return s12s6_reshape

    def energy(self, xi_space):

        '''
        energy function to get potential energy

        Returns
        ----------
        term : torch.tensor
                total potential energy of q state each sample
                shape is [nsamples]
        '''

        xi_state = xi_space.get_q()
        boxsize = xi_space.get_boxsize()

        nsamples, nparticle, DIM  = xi_state.shape

        a12 = (4 * self._epsilon * pow(self._sigma, 12)) / pow(boxsize, 12)
        a6 = (4 * self._epsilon * pow(self._sigma, 6)) / pow(boxsize, 6)

        _, d = xi_space.paired_distance_reduced(xi_state, nparticle, DIM)
        # d.shape is [nsamples, nparticle, (nparticle - 1 )]

        s12 = 1 / pow(d,12)
        s6  = 1 / pow(d,6)

        term_dim = (a12 * torch.sum(s12, dim=-1) - a6 * torch.sum(s6, dim=-1))
        # term_dim.shape is [nsamples, nparticle]

        term =  torch.sum(term_dim, dim=-1) * 0.5
        # term.shape is [nsamples]

        return term

    def evaluate_derivative_q(self,xi_space):

        '''
        evaluate_derivative_q function to get dUdq

        Returns
        ----------
        dphidxi : torch.tensor
                derivative of potential energy wrt q state each sample
                shape is [nsamples, nparticle, DIM]
        '''

        xi_state = xi_space.get_q()
        boxsize = xi_space.get_boxsize()

        nsamples, nparticle, DIM  = xi_state.shape

        a12 = (4 * self._epsilon * pow(self._sigma, 12)) / pow(boxsize, 13)
        a6 = (4 * self._epsilon * pow(self._sigma, 6)) / pow(boxsize, 7)

        delta_xi, d = xi_space.paired_distance_reduced(xi_state,nparticle,DIM)
        # delta_xi.shape is [nsamples, nparticle, (nparticle - 1), DIM]
        # d.shape is [nsamples, nparticle, (nparticle - 1 )]

        d = torch.unsqueeze(d,dim =-1)
        # d.shape is [nsamples, nparticle, (nparticle - 1 ), 1]

        s12 = -12 * (delta_xi) / pow(d,14)
        s6  = -6 * (delta_xi) / pow(d,8)

        dphidxi = a12*torch.sum(s12, dim=2) - a6*torch.sum(s6, dim=2) # np.sum axis=2 j != k ( nsamples-1)
        # dphidxi.shape is [nsamples, nparticle, DIM]
        # print(dphidxi)
        return dphidxi


    def evaluate_second_derivative_q(self,xi_space):

        xi_state = xi_space.get_q()
        boxsize = xi_space.get_boxsize()
        d2phidxi2_append = []

        nsamples, nparticle, DIM  = xi_state.shape
        d2phidxi2 = torch.zeros((nsamples, nparticle * DIM, nparticle * DIM)) # second derivative terms of nsamples
        d2phidxi_lk = torch.zeros((2,2))


        a12 = (4 * self._epsilon * pow(self._sigma, 12)) / pow(boxsize, 14)
        a6 = (4 * self._epsilon * pow(self._sigma, 6)) / pow(boxsize, 8)

        for z in range(nsamples):

            d2phidxi2_ = torch.empty((0, nparticle * DIM))

            delta_xi, d = xi_space.paired_distance_reduced(xi_state[z],nparticle,DIM)
            d = torch.unsqueeze(d,dim=2)

            s12_same_term = 1. / pow(d,14)
            s12_lxkx_lyky = (-14) * torch.pow(delta_xi,2) / torch.pow(d,2)
            s12_lxky_lykx = 1. / pow(d,16)

            s6_same_term = 1. / pow(d,8)
            s6_lxkx_lyky = (-8) * torch.pow(delta_xi,2) / torch.pow(d,2)
            s6_lxky_lykx = 1. / pow(d,10)

            for l in range(nparticle):
                j = 0
                for k in range(nparticle):

                    if l == k:
                        d2phidxi_lxkx = a12 *(-12)* (torch.sum(s12_same_term[k] * torch.unsqueeze(s12_lxkx_lyky[k,:,0],dim=-1) + s12_same_term[k],dim=0)) \
                                        - a6 *(-6)* (torch.sum(s6_same_term[k] * torch.unsqueeze(s6_lxkx_lyky[k,:,0],dim=-1) + s6_same_term[k],dim=0))

                        d2phidxi_lxky = a12 * (-12)*(-14)*(torch.sum(s12_lxky_lykx[k]*torch.unsqueeze(delta_xi[k,:,0],dim=-1)*torch.unsqueeze(delta_xi[k,:,1],dim=-1),dim=0))  \
                                        - a6 * (-6)*(-8)*(torch.sum(s6_lxky_lykx[k]*torch.unsqueeze(delta_xi[k,:,0],dim=-1)*torch.unsqueeze(delta_xi[k,:,1],dim=-1),dim=0))

                        d2phidxi_lykx = a12 * (-12)*(-14)*(torch.sum(s12_lxky_lykx[k]*torch.unsqueeze(delta_xi[k,:,0],dim=-1)*torch.unsqueeze(delta_xi[k,:,1],dim=-1),dim=0)) \
                                        - a6 * (-6)*(-8)*(torch.sum(s6_lxky_lykx[k]*torch.unsqueeze(delta_xi[k,:,0],dim=-1)*torch.unsqueeze(delta_xi[k,:,1],dim=-1),dim=0))

                        d2phidxi_lyky = a12 *(-12)* (torch.sum(s12_same_term[k] * torch.unsqueeze(s12_lxkx_lyky[k,:,1],dim=-1) + s12_same_term[k],dim=0)) \
                                        - a6 *(-6)* (torch.sum(s6_same_term[k] * torch.unsqueeze(s6_lxkx_lyky[k,:,1],dim=-1) + s6_same_term[k],dim=0))

                        d2phidxi_lk = torch.tensor((d2phidxi_lxkx[0], d2phidxi_lxky[0], d2phidxi_lykx[0], d2phidxi_lyky[0])).reshape(2, 2)
                        print('l=k d2phidxi_lk',d2phidxi_lk)

                    if l != k:
                        print('j',j)

                        d2phidxi_lxkx = - a12 *(-12)* s12_same_term[l][j] *( s12_lxkx_lyky[l][j][0] + 1) \
                                        + a6 *(-6)* s6_same_term[l][j] * ( s6_lxkx_lyky[l][j][0] + 1)

                        d2phidxi_lxky = - a12 * (-12)*(-14) * (s12_lxky_lykx[l][j] * delta_xi[l][j][0] * delta_xi[l][j][1]) \
                                        + a6 * (-6)*(-8) * (s6_lxky_lykx[l][j] * delta_xi[l][j][0] * delta_xi[l][j][1])

                        d2phidxi_lykx = - a12 * (-12)*(-14)*(s12_lxky_lykx[l][j] * delta_xi[l][j][0]* delta_xi[l][j][1]) \
                                        + a6 * (-6)*(-8)* (s6_lxky_lykx[l][j] * delta_xi[l][j][0] * delta_xi[l][j][1])

                        d2phidxi_lyky = - a12 *(-12)* s12_same_term[l][j] * ( s12_lxkx_lyky[l][j][1]  + 1) \
                                        + a6 *(-6)* s6_same_term[l][j]  * ( s6_lxkx_lyky[l][j][1] + 1)

                        d2phidxi_lk = torch.tensor((d2phidxi_lxkx[0],d2phidxi_lxky[0],d2phidxi_lykx[0],d2phidxi_lyky[0])).reshape(2,2)
                        print('l != k d2phidxi_lk',d2phidxi_lk)

                        j = j + 1
                    d2phidxi2_append.append(d2phidxi_lk)

                if k == nparticle - 1:
                    temp = torch.stack(d2phidxi2_append,dim=0)
                    temp = temp.permute((1,0,2)).reshape(2, nparticle * DIM )
                    print('reshape',temp)
                    d2phidxi2_append = []

                d2phidxi2_ = torch.cat((d2phidxi2_,temp),dim=0)

            d2phidxi2[z] = d2phidxi2_
        print('d2phidxi2', d2phidxi2)

        return d2phidxi2
