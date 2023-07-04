from utils.pbc import pbc
from utils.utils import assert_nan
import numpy as np

class velocity_verletx:

    neval = 3
    nnet  = neval*2

    def __init__(self,force_function):
        self.force_function = force_function
        self.modef = force_function.get_mode()

        assert(self.modef == 'ff'),'hf mode not implemented in velocity_verlet3'

    def one_step(self,q_list0,p_list0,l_list):

        p_input0 = self.force_function.prepare_p_input(q_list0,p_list0,l_list)
        q_input0 = self.force_function.prepare_q_input(0,q_list0,p_list0,l_list)

        p_input_list = [p_input0]
        q_input_list = [q_input0]
       
        p_list_pre = p_list0
        q_list_pre = q_list0

        for s in range(velocity_verletx.neval):

            p_list_cur = p_list_pre + self.force_function.eval(q_input_list,p_input_list)
            p_input_next = self.force_function.prepare_p_input(q_list_pre,p_list_cur,l_list)
            p_input_list.append(p_input_next)

            tau = self.force_function.get_tau(2*s+1)
            q_list_cur = q_list_pre + tau*p_list_cur + self.force_function.eval(q_input_list,p_input_list)
            #q_list_cur = q_list_pre + self.force_function.eval(q_input_list,p_input_list)
            q_list_cur = pbc(q_list_cur,l_list)
            net_idx = s
            q_input_next = self.force_function.prepare_q_input(net_idx,q_list_cur,p_list_cur,l_list)
            q_input_list.append(q_input_next)
            p_list_pre = p_list_cur
            q_list_pre = q_list_cur

        assert_nan(p_list_cur)
        assert_nan(q_list_cur)
        return q_list_cur,p_list_cur,l_list


    def nsteps(self,q_list,p_list,l_list,n_chain):

        nsamples,nparticles,_ = q_list.shape
        self.force_function.make_mask(nsamples,nparticles)
        for chain in range(n_chain):
            q_list,p_list,l_list = self.one_step(q_list,p_list,l_list)
        return q_list,p_list,l_list
