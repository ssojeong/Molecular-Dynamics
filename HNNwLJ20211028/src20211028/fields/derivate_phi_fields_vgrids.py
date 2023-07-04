import torch
from fields.hex_grids import hex_grids

class derivate_phi_fields_vgrids:

    ''' phi_fields class to help calculate phi fields on grids '''

    _obj_count = 0

    def __init__(self, noML_hamiltonian, dgrid, ngrids, dphi_maxcut=108.35):  # rij=0.9 -> dphi ~ 108.35
        '''
        Parameters
        ----------
        noML_hamiltonian : noML obj
        dgrid   : distance btw grids
        ngrids  : each particle has 18 grids
        maxcut  : threshold for potential energy
        mincut  : -6 and then give margin -2 = -8 <= each grid can have nearest particles maximum 6
        '''

        derivate_phi_fields_vgrids._obj_count += 1
        assert(derivate_phi_fields_vgrids._obj_count == 1), type(self).__name__ + ' has more than one object'

        self.ngrids = ngrids
        terms = noML_hamiltonian.get_terms()
        self.lennard_jones = terms[0]
        self.epsilon = self.lennard_jones.epsilon
        self._dphi_maxcut = dphi_maxcut

        self.hex_grids = hex_grids(dgrid, ngrids)
        self.dpair_pbc = self.lennard_jones.dpair_pbc
        print('phi_fields initialized : ngrids',ngrids, 'dgrid', dgrid, ' dphi maxcut ',dphi_maxcut)

    def grids_fixed(self, phase_space):
        ''' function to fix initial grids when q, p state move from initial time step to next time step

        Parameters
        ----------
        phase_space : contains q_list, p_list as input
                q_list shape is [nsamples, nparticle, DIM]

        '''
        self.grids_list = self.hex_grids.make_grids(phase_space)
        # shape is [nsamples, nparticle, ngrids, DIM=(x,y)]

        nsamples,nparticle,ngrids,DIM = self.grids_list.shape

        # self.hex_grids.show_grids_nparticles(nsamples, phase_space.get_q()[0], phase_space.get_boxsize())  # show about one sample

        self.grids_list = self.grids_list.reshape(-1,nparticle*ngrids,DIM)
        # shape is [nsamples, nparticle*ngrids, DIM=(x,y)]


    def gen_derivative_phi_fields(self, phase_space):

        nsamples, nparticle, DIM = phase_space.get_q().shape

        # # check grids fix after integrate at small time step
        # self.hex_grids.show_grids_nparticles(self.grids_list, phase_space.get_q()[0], phase_space.get_boxsize())  # show about one sample
        # # print('grids list',self.grids_list)

        _derivative_phi_field = self.lennard_jones.derivate_phi_fields(phase_space, self.grids_list)
        # shape is [ nsamples, nparticle*ngrids, DIM ]

        if __debug__: # any elements have nan then become true ,so that get error
            assert ((torch.isnan(_derivative_phi_field)).any() == False ), 'error .. any phi_fields nan ...'

        # derivative phi-fields clipping
        mask1 = _derivative_phi_field > self._dphi_maxcut
        if mask1.any() == True:
            _derivative_phi_field[mask1] = self._dphi_maxcut

        mask2 = _derivative_phi_field < -self._dphi_maxcut
        if mask2.any() == True:
            _derivative_phi_field[mask2] = -self._dphi_maxcut

        _gen_derivative_phi_field = _derivative_phi_field.reshape((nsamples, nparticle, self.ngrids * DIM))
        # shape is [ nsamples, nparticle, ngrids*DIM ]

        return _gen_derivative_phi_field

    def v_ngrids(self, phase_space):
        '''
        function to get velocity each grid at particle position as center

        Parameters
        ----------
        phase_space : torch.tensor
                p_list.shape is [nsamples, nparticle, DIM]
                boxsize.shape is [nsamples, nparticle, DIM]
        '''

        p_list = phase_space.get_p()
        boxsize =phase_space.get_l_list()

        boxsize = torch.unsqueeze(boxsize, dim=2)
        boxsize = boxsize.repeat_interleave(self.grids_list.shape[1],dim=2)
        # boxsize.shape is [nsamples, nparticle, nparticle * ngrids, DIM]

        nsamples, nparticle, DIM = p_list.shape

        _, d = self.dpair_pbc.paired_grid_q(phase_space, self.grids_list, boxsize)
        # d.shape is [nsamples, nparticle, nparticle * ngrids]

        w_thrsh = self.weight_thrsh(d, boxsize)
        # w_thrsh.shape is [nsamples, nparticle, nparticle * ngrids]

        w_thrsh = torch.unsqueeze(w_thrsh, dim=-1)
        # w_thrsh.shape is [nsamples, nparticle, nparticle * ngrids, 1]

        p_list = torch.unsqueeze(p_list, dim=2)
        # p_list.shape is [nsamples, nparticle, 1, DIM]

        wp = w_thrsh * p_list
        # w * m * velocity ; m=1
        # wp.shape [nsamples, nparticle, nparticle * ngrids, DIM]

        wp_nume = torch.sum(wp, dim=1)
        # wp_nume.shape [nsamples,  nparticle * ngrids, DIM]
        wp_deno = torch.sum(w_thrsh, dim=1)
        # wp_deno.shape is [nsamples, nparticle * ngrids, 1]

        p_ngrids = wp_nume / wp_deno
        # p_grids.shape [nsamples,  nparticle * ngrids, DIM]

        p_ngrids = p_ngrids.reshape((nsamples, nparticle, self.ngrids, DIM))
        # p_ngrids.shape [nsamples, npartice, grids, DIM]
        # p_list.shape is [nsamples, nparticle, 1, DIM]

        # relative momentum
        relative_p = p_ngrids - p_list
        # relative_p.shape [nsamples, npartice, grids, DIM]

        relative_p = relative_p.reshape((nsamples, nparticle, self.ngrids*DIM))
        # relative_p.shape [nsamples, npartice, grids*DIM]

        return relative_p

    def weight_thrsh(self, d, boxsize):
        '''
        function to obtain weights of particle and ignore the large distance btw pairs of particles

        Parameters
        ----------
        d : torch.tensor [nsamples, nparticle, nparticle * ngrids]
        boxsize : torch.tensor [nsamples, nparticle, nparticle * ngrids, DIM]
        '''
        boxsize = torch.mean(boxsize,dim=-1)
        # boxsize.shape is [nsamples, nparticle, nparticle * ngrids]

        w = torch.zeros(d.shape)
        # w.shape is [nsamples, nparticle, nparticle * ngrids]

        max_d = torch.sqrt(2 * (0.5 * boxsize) ** 2)

        # if d=sqrt(r^2) < d_thrsh, w = 1/d^2
        thrsh_indx = d < 0.25*max_d

        w[thrsh_indx] = 1 / (d[thrsh_indx] )
        # w.shape is [nsamples, nparticle, nparticle * ngrids]

        return w