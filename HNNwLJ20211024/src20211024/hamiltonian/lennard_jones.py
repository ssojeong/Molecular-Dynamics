from hamiltonian.LJ_term import LJ_term
from phase_space.phase_space import phase_space

class lennard_jones:
    '''lennard_jones class for dimensionless potential and derivative '''

    _obj_count = 0

    def __init__(self,sigma = 1,epsilon = 1):

        lennard_jones._obj_count += 1
        assert (lennard_jones._obj_count == 1),type(self).__name__ + " has more than one object"

        self.epsilon = epsilon
        self.sigma = sigma
        self.phi = LJ_term(self.epsilon, self.sigma)
        self.dpair_pbc = self.phi.dpair_pbc

        print('lennard_jones.py call potential')
        self._name = 'Lennard Jones Potential'

        # make a local copy of phase space, so as to separate dimensionless and non-dimensionless
        # copies of phase space
        self.dimensionless_phase_space = phase_space()

        print('lennard_jones initialized: sigma ',sigma,' epsilon ',epsilon)

    def dimensionless(self, phase_space):
        ''' For computation convenience, rescale the system so that boxsize is 1
        parameter
        ------------
        phase space : contains q_list, p_list and, boxsize
        '''

        q_state = phase_space.get_q()
        # q_state shape is [nsamples, nparticle, DIM]

        boxsize = phase_space.get_boxsize()

        q_state = q_state / boxsize
        self.dimensionless_phase_space.set_q(q_state)
        self.dimensionless_phase_space.set_boxsize(boxsize)

        return self.dimensionless_phase_space

    def dimensionless_grids(self, grid_list, boxsize): 
        ''' For computation convenience, rescale the system so that grid list is 1

        parameter
        ------------
        grid_list.shape is [nsamples, nparticle*grids18, DIM=(x coord, y coord)]
        '''

        dimensionless_grids_list = grid_list /  boxsize

        return dimensionless_grids_list

    def phi_fields(self, phase_space, grid_list): 
        ''' phi fields function to get phi fields each grid point

        parameters :
        grid_list : shape is [nsamples, nparticle * ngrids, DIM=(x,y)]
        '''

        xi_space = self.dimensionless(phase_space)
        # shape is [nsamples, nparticle, DIM]

        boxsize = phase_space.get_boxsize()

        dimensionless_grids_list = self.dimensionless_grids(grid_list, boxsize)
        # shape is [nsamples, nparticle * ngrids, DIM=(x,y)]

        return self.phi.phi_fields(xi_space, dimensionless_grids_list)


    def derivate_phi_fields(self, phase_space, grid_list):
        ''' dphi fields function to get derivate phi fields each grid point

        parameters :
        grid_list : shape is [nsamples, nparticle * ngrids, DIM=(x,y)]
        '''

        xi_space = self.dimensionless(phase_space)
        # shape is [nsamples, nparticle, DIM]

        boxsize = phase_space.get_boxsize()

        dimensionless_grids_list = self.dimensionless_grids(grid_list, boxsize)
        # shape is [nsamples, nparticle * ngrids, DIM=(x,y)]

        return self.phi.derivate_phi_fields(xi_space, dimensionless_grids_list)


    def energy(self, phase_space):
        ''' energy function to get potential energy '''

        xi_space = self.dimensionless(phase_space)
        return self.phi.energy(xi_space)

    def evaluate_derivative_q(self, phase_space):
        ''' evaluate_derivative_q function to get dUdq '''

        xi_space = self.dimensionless(phase_space)
        dphidq = self.phi.evaluate_derivative_q(xi_space)
        return dphidq

    def evaluate_second_derivative_q(self, phase_space):
        xi_space = self.dimensionless(phase_space)
        d2phidq2 = self.phi.evaluate_second_derivative_q(xi_space)
        return d2phidq2
