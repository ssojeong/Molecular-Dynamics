import torch
from HNN.HNN_base                   import HNN_base
from fields.derivate_phi_fields     import derivate_phi_fields
import time

class fields_HNN(HNN_base):
    ''' fields_HNN class to learn dHdq and then combine with nodHdq '''

    _obj_count = 0

    def __init__(self, fields_MLP1, fields_MLP2, dgrid, ngrid, on_off_noML):

        super(fields_HNN,self).__init__(fields_MLP1, fields_MLP2) # python 2 ; python 3 super().__init__()

        fields_HNN._obj_count += 1
        assert (fields_HNN._obj_count == 1),type(self).__name__ + " has more than one object"

        self.derivate_phi_fields  = derivate_phi_fields(super(), dgrid, ngrid)
        self.ngrid              = ngrid
        self.on_off_noML        = on_off_noML

        self.tau_short          = None # check give tau or not
        self.tau_long           = None # check give tau or not

        print('fields_HNN initialized ')
    # ===================================================
    def set_tau_short(self, tau_short): 
        self.tau_short = tau_short
    # ===================================================
    def dHdq1(self, phase_space):
        ''' use this to update p in first step of velocity verlet
        set ml_dhdq1 in phase space and can use this to get configuration
        before crash in crash_chkr '''
        # print('dhdq1')
        noml_dhdq, ml_dhdq1 = self.dHdq_all(phase_space, self.net1)
        Rx, Ry = self.calculate_RxRy_ratio(noml_dhdq, ml_dhdq1)
        self.sum_RxRy_dhdq1(Rx, Ry)
        phase_space.set_ml_dhdq1(ml_dhdq1)
        return noml_dhdq, ml_dhdq1
    # ===================================================
    def dHdq2(self, phase_space):
        ''' use this to update p in third step of velocity verlet
        set ml_dhdq2 in phase space and can use this to get configuration
        before crash in crash_chkr '''
        # print('dhdq2')
        noml_dhdq,ml_dhdq2 = self.dHdq_all(phase_space, self.net2)
        Rx, Ry = self.calculate_RxRy_ratio(noml_dhdq, ml_dhdq2)
        self.sum_RxRy_dhdq2(Rx, Ry)
        phase_space.set_ml_dhdq2(ml_dhdq2)
        return noml_dhdq, ml_dhdq2
    # ===================================================
    def dHdq_all(self, phase_space, net): # nochange
        ''' function to calculate dHdq = noML_dHdq + residual ML_dHdq

        Parameters
        ----------
        phase_space : contains q_list, p_list as input
                q_list shape is [nsamples, nparticle, DIM]
        net         : pass fields_hnn

        Returns
        ----------
        corrected_dHdq : torch.tensor
                shape is [nsamples,nparticle,DIM]

        '''
        nsamples, nparticle, DIM = phase_space.get_q().shape

        if self.on_off_noML == 'on':
            noML_dHdq = self.dHdq(phase_space) # super().dHdq(phase_space)
            # noML_dHdq shape is [nsamples, nparticle, DIM]
        else:
            noML_dHdq = torch.zeros([nsamples, nparticle, DIM])  # not use noML in f-hn
            # noML_dHdq shape is [nsamples, nparticle, DIM]

        x = self.make_dphi_fields_tau(phase_space, self.tau_long)
        # x.shape = [ nsamples, nparticle,  ngrids * DIM + 1]

        x = torch.reshape(x, (nsamples * nparticle, 2 * self.ngrid*DIM + 1))
        # x.shape = [ nsamples*nparticle, ngrids * DIM + ngrids * DIM + 1]

        start = time.time()
        out = net(x)
        # out.shape = [nsamples*nparticle, 2]
        end = time.time()
        self.dt += (end-start)

        predict = torch.reshape(out, (nsamples, nparticle, DIM))
        # predict.shape = [nsamples, nparticle, DIM=(x,y)]

        return noML_dHdq, predict

    # ===================================================
    def make_dphi_fields_tau(self, phase_space, tau_long):
        ''' function to make derivate fields for feeding into nn

        Parameters
        ----------
        phase_space : contains q_list, p_list as input
                q_list shape is [nsamples, nparticle, DIM]
        tau_long  : use to concatenate two fields and tau long

        Returns
        ----------
        dphi25.shape : shape is [ nsamples, nparticle,  ngrids * DIM + ngrids * DIM + 1]
        '''

        q_list = phase_space.get_q()
        nsamples, nparticle, DIM = q_list.shape

        self.derivate_phi_fields.grids_fixed(phase_space)
        p_ngrids = self.derivate_phi_fields.v_ngrids(phase_space)
        # p_ngrids.shape is [ nsamples, nparticle, ngrids * DIM ]
        ndphi_1 = self.derivate_phi_fields.gen_derivative_phi_fields(phase_space)
        # ndphi_1.shape is [ nsamples, nparticle, ngrids * DIM ]

        tau_tensor = torch.zeros([nsamples, nparticle, 1])
        tau_tensor.fill_(tau_long * 0.5)  # tau_tensor take them tau/2

        dphi25 = torch.cat((ndphi_1, p_ngrids, tau_tensor), dim=-1)
        # n1phi.shape = shape is [ nsamples, nparticle,  ngrids * DIM + ngrids * DIM + 1]

        return dphi25
    # ===================================================

    # def print_grad(self):
    #     self.net1.print_grad()
    #     self.net2.print_grad()


