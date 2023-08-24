import torch
import numpy as np
import torch.nn as nn
from utils.mydevice import mydevice

from utils.pbc import pbc
from utils.utils import assert_nan

class LLUF_MD(nn.Module):

    def __init__(self,prepare_data, LLUF_update_p, LLUF_update_q,t_init=1, nnet=1):
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
    def one_step(self,q_input_list,p_input_list,q_pre,p_pre,l_list):

        # q_input_list [phi0,phi1,phi2,...]  -- over time point
        # phi0.shape = [nsamples*nparticles, ngrids*DIM]
        # p_input_list [pi0,pi1,pi2,...]

        p_cur = p_pre + self.LLUF_update_p(q_input_list,p_input_list, q_pre) # SJ coord

        p_input_cur = self.prepare_data.prepare_p_feature_input(q_pre,p_cur,l_list)
        p_input_list.append(p_input_cur)
        p_input_list.pop(0)  # remove first element

        q_cur = q_pre + torch.abs(self.tau) * p_cur + self.LLUF_update_q(q_input_list,p_input_list, q_pre) # SJ coord
        q_cur = pbc(q_cur,l_list)

        q_input_next = self.prepare_data.prepare_q_feature_input(q_cur,l_list)
        q_input_list.append(q_input_next)
        q_input_list.pop(0)

        assert_nan(p_cur)
        assert_nan(q_cur)
        return q_input_list,p_input_list,q_cur,p_cur,l_list


    def nsteps(self,q_input_list,p_input_list,q_pre,p_pre,l_list,window_sliding):

        #assert(n_chain==1),'MD/velocity_verletx,py: error only n_chain = 1 is implemented '

        # our mbpw-net model chain up to predict the new configuration for n-times
        for ws in range(window_sliding):

            q_input_list,p_input_list,q_cur,p_cur,l_list = \
                                  self.one_step(q_input_list,p_input_list,q_pre,p_pre,l_list)
            q_pre = q_cur
            p_pre = p_cur

        return q_input_list,p_input_list,q_cur,p_cur,l_list
