import torch
from hamiltonian.hamiltonian            import hamiltonian
from hamiltonian.lennard_jones          import lennard_jones
from hamiltonian.kinetic_energy         import kinetic_energy
from fields.phi_fields                  import phi_fields

class fields_HNN(hamiltonian):
    ''' pairwise_HNN class to learn dHdq and then combine with nodHdq '''

    _obj_count = 0

    def __init__(self, fields_cnn, integrator, interpolator):

        fields_HNN._obj_count += 1
        assert (fields_HNN._obj_count == 1),type(self).__name__ + " has more than one object"

        super(fields_HNN,self).__init__() # python 2 ; python 3 super().__init__()

        # # append term to calculate dHdq
        super().append(lennard_jones())
        super().append(kinetic_energy())

        self.fields_cnn         = fields_cnn # network for dHdq
        self.interpolator       = interpolator
        self.integrator         = integrator
        self.phi_fields         = phi_fields(fields_cnn.get_gridL(), super())

        self.tau_cur      = None # check give tau or not

        print('fields_HNN initialized ')

    # ===================================================
    def set_tau(self, tau_cur): # nochange
        self.tau_cur = tau_cur

    # ===================================================
    def get_netlist(self): # nochange
        return [self.fields_cnn]
    # ===================================================

    def net_parameters(self): # nochange
        ''' To give multiple parameters to the optimizer,
            concatenate lists of parameters from two models '''

        return self.fields_cnn.parameters()

    # ===================================================
    def train(self): # nochange
        '''pytorch network for training'''
        self.fields_cnn.train()

    # ===================================================
    def eval(self): # nochange
        ''' pytorch network for eval '''
        self.fields_cnn.eval()

    # ===================================================
    def dHdq1(self, phase_space): # nochange
        # use this to update p in first step of velocity verlet
        return self.dHdq_all(phase_space, self.fields_cnn)
    # ===================================================
    def dHdq2(self, phase_space): # nochange
        # use this to update p in third step of velocity verlet
        return self.dHdq_all(phase_space, self.fields_cnn)
    # ===================================================
    def dHdq_all(self, phase_space, net): # nochange

        ''' function to calculate dHdq = noML_dHdq + residual ML_dHdq

        Parameters
        ----------
        phase_space : contains q_list, p_list as input
                q_list shape is [nsamples, nparticle, DIM]
        net         : pass fields_cnn

        Returns
        ----------
        corrected_dHdq : torch.tensor
                shape is [nsamples,nparticle,DIM]

        '''
        q_list = phase_space.get_q()

        noML_dHdq = super().dHdq(phase_space)
        # noML_dHdq shape is [nsamples, nparticle, DIM]

        x = self.make_fields(phase_space, self.integrator, self.tau_cur)
        # x.shape = [nsamples, nchannels=2, gridLx, gridLy]

        predict = net(x)
        # predict.shape = [nsamples, gridLx, gridLy, DIM=(fx,fy)]
        # # SJ remember to uncomment later
        # predict_force = self.interpolator.inverse_distance_interpolator(predict, q_list)
        #
        # corrected_dHdq.shape = [nsamples, nparticle, DIM]
        # corrected_dHdq = noML_dHdq + predict_force

        corrected_dHdq = noML_dHdq # for testing if can run

        return corrected_dHdq

    # ===================================================

    def make_fields(self, phase_space, integrator, tau_cur):

        q_list_original = phase_space.get_q()
        p_list_original = phase_space.get_p()

        img1 = self.phi_fields.gen_phi_fields(phase_space)
        # img1.shape is [ nsamples, gridL, gridL ]

        # self.phi_fields.show_gridimg(img1, ' at time t')

        qp_list, crash_idx = integrator.one_step( super(), phase_space, tau_cur)
        # qp_list shape, [nsamples, (q,p)=2, nparticle, DIM]

        phase_space.set_q(qp_list[:,0,:,:])
        phase_space.set_p(qp_list[:,1,:,:])

        img2 = self.phi_fields.gen_phi_fields(phase_space)
        # img2 shape is [ nsamples, gridL, gridL ]

        # self.phi_fields.show_gridimg(img2, ' at time t+tau')

        # copy back to original
        phase_space.set_q(q_list_original)
        phase_space.set_p(p_list_original)

        img = torch.stack((img1,img2),dim=1)
        # img.shape = [nsamples,nchannels=2,gridL,gridL]

        return img