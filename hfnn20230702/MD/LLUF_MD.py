from utils.pbc import pbc
from utils.utils import assert_nan
import numpy as np

class LLUF_MD:

    def __init__(self,force_function):
        self.force_function = force_function
        print(' velocity verletx ')

        #assert(self.modef == 'ff'),'hf mode not implemented in velocity_verlet3'

    # q_input_list [(phi0,dq0),(phi1,dq1),(phi2,dq2),...] -- over time points
    # p_input_list [(pi0,dp0),(pi1,dp1),(pi2,dp2),...]
    def one_step(self,q_input_list,p_input_list,q_pre,p_pre,l_list):

        use_p_netid = 0
        use_q_netid = 1
        prepare_q_input_netid = 0

        # q_input_list [(phi0,dq0),(phi1,dq1),(phi2,dq2),...]  -- over time point
        # phi0.shape = [nsamples*nparticles, ngrids*DIM]
        # p_input_list [(pi0,dp0),(pi1,dp1),(pi2,dp2),...]
        p_cur = p_pre + self.force_function.eval(use_p_netid,q_input_list,p_input_list, q_pre) # SJ coord

        p_input_cur = self.force_function.prepare_p_input(q_pre,p_cur,l_list)
        p_input_list.append(p_input_cur)
        p_input_list.pop(0)  # remove first element

        tau = self.force_function.get_tau(1)
        q_cur = q_pre + tau*p_cur + self.force_function.eval(use_q_netid,q_input_list,p_input_list, q_pre) # SJ coord
        q_cur = pbc(q_cur,l_list)

        q_input_next = self.force_function.prepare_q_input(prepare_q_input_netid,q_cur,p_cur,l_list)
        q_input_list.append(q_input_next)
        q_input_list.pop(0)

        assert_nan(p_cur)
        assert_nan(q_cur)
        return q_input_list,p_input_list,q_cur,p_cur,l_list


    def nsteps(self,q_input_list,p_input_list,q_pre,p_pre,l_list,n_chain):

        #assert(n_chain==1),'MD/velocity_verletx,py: error only n_chain = 1 is implemented '

        # our mbpw-net model chain up to predict the new configuration for n-times
        for chain in range(n_chain):

            q_input_list,p_input_list,q_cur,p_cur,l_list = \
                                  self.one_step(q_input_list,p_input_list,q_pre,p_pre,l_list)
            q_pre = q_cur
            p_pre = p_cur

        return q_input_list,p_input_list,q_cur,p_cur,l_list
