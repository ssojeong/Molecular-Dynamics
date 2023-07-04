from utils.pbc import pbc
from utils.utils import assert_nan
import numpy as np

class velocity_verlet3:

    neval = 6

    def __init__(self,force_function):
        self.force_function = force_function
        self.modef = force_function.get_mode()

        assert(self.modef == 'ff'),'hf mode not implemented in velocity_verlet3'

    def one_step(self,q_list0,p_list0,l_list):

        # update p0 -> p1
        p_input0 = self.force_function.prepare_p_input(q_list0,p_list0,l_list)
        q_input0 = self.force_function.prepare_q_input(0,q_list0,p_list0,l_list)
        p_input_list = [p_input0]
        q_input_list = [q_input0]
        p_list1 = p_list0 + self.force_function.eval(q_input_list,p_input_list)

        # update q0 -> q1
        p_input1 = self.force_function.prepare_p_input(q_list0,p_list1,l_list) ##### note, use q_list0 instead of q_list1
        p_input_list.append(p_input1)
        q_list1 = q_list0 + self.force_function.eval(q_input_list,p_input_list)
        #q_list1.retain_grad()
        q_list1 = pbc(q_list1,l_list)

        # update p1 -> p2
        q_input1 = self.force_function.prepare_q_input(1,q_list1,p_list1,l_list)
        q_input_list.append(q_input1)
        p_list2 = p_list1 + self.force_function.eval(q_input_list,p_input_list)

        #assert_nan(p_list2)
        #assert_nan(q_list1)
        #return q_list1,p_list2,l_list

        # update q1 -> q2
        p_input2 = self.force_function.prepare_p_input(q_list1,p_list2,l_list) ##### note, use q_list1 instead of q_list2
        p_input_list.append(p_input2)
        q_list2 = q_list1 + self.force_function.eval(q_input_list,p_input_list)
        #q_list2.retain_grad()
        q_list2 = pbc(q_list2,l_list)

        # update p2 -> p3
        q_input2 = self.force_function.prepare_q_input(2,q_list2,p_list2,l_list)
        q_input_list.append(q_input2)
        p_list3 = p_list2 + self.force_function.eval(q_input_list,p_input_list)

        # update q2 -> q3
        p_input3 = self.force_function.prepare_p_input(q_list2,p_list3,l_list) ##### note, use q_list2 instead of q_list3
        p_input_list.append(p_input3)
        q_list3 = q_list2 + self.force_function.eval(q_input_list,p_input_list)
        #q_list3.retain_grad()
        q_list3 = pbc(q_list3,l_list)

        assert_nan(p_list3)
        assert_nan(q_list3)
        return q_list3,p_list3,l_list


    def nsteps(self,q_list,p_list,l_list,n_chain):

        nsamples,nparticles,_ = q_list.shape
        self.force_function.make_mask(nsamples,nparticles)
        for chain in range(n_chain):
            q_list,p_list,l_list = self.one_step(q_list,p_list,l_list)
        return q_list,p_list,l_list
