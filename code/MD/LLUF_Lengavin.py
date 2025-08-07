import torch
import numpy as np
import torch.nn as nn
from utils.mydevice import mydevice
from hamiltonian.thermostat import thermostat_ML

from utils.pbc import pbc
from utils.utils import assert_nan

class LLUF_Lengavin(nn.Module):

    def __init__(self,prepare_data, LLUF_update_p, LLUF_update_q, t_init=1, nnet=1):
        super().__init__()

        self.prepare_data = prepare_data
        self.LLUF_update_p = LLUF_update_p
        self.LLUF_update_q = LLUF_update_q
        self.tau_init = np.random.rand(nnet) * t_init  # change form 0.01 to 0.001

        self.tau = nn.Parameter(torch.tensor(self.tau_init, device=mydevice.get()))
        print(' velocity verletx ')

        #assert(self.modef == 'ff'),'hf mode not implemented in velocity_verlet3'

    # q_input_list [phi0,phi1,phi2,...] -- over time points
    # p_input_list [pi0,pi1,pi2,...]
    def one_step(self,q_input_list,p_input_list,q_pre,p_pre,l_list, gamma=0, temp=0, tau_long=0.1):

        # q_input_list [phi0,phi1,phi2,...]  -- over time point
        # phi0.shape = [nsamples*nparticles, ngrids*DIM]
        # p_input_list [pi0,pi1,pi2,...]

        q_cur = q_pre  + p_pre * self.tau + self.LLUF_update_q(q_input_list, p_input_list, q_pre)
        q_cur = pbc(q_cur, l_list)

        q_input_next = self.prepare_data.prepare_q_feature_input(q_cur, l_list)
        q_input_list.append(q_input_next)
        q_input_list.pop(0)

        p_cur = p_pre + self.LLUF_update_p(q_input_list,p_input_list, q_cur) # SJ coord

        p_cur = thermostat_ML(p_cur, gamma, temp, tau_long)

        p_input_cur = self.prepare_data.prepare_p_feature_input(q_cur,p_cur,l_list)
        p_input_list.append(p_input_cur)
        p_input_list.pop(0)  # remove first element

        assert_nan(p_cur)
        assert_nan(q_cur)
        return q_input_list,p_input_list,q_cur,p_cur,l_list


    def nsteps(self,q_input_list,p_input_list,q_pre,p_pre,l_list):

        #assert(n_chain==1),'MD/velocity_verletx,py: error only n_chain = 1 is implemented '

        # our mbpw-net model chain up to predict the new configuration for n-times
        q_input_list,p_input_list,q_cur,p_cur,l_list = \
                                  self.one_step(q_input_list,p_input_list,q_pre,p_pre,l_list)

        return q_input_list,p_input_list,q_cur,p_cur,l_list
