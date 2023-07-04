from utils.pbc import pbc
from utils.utils import assert_nan
import numpy as np

class velocity_verlet2:

    def __init__(self,force_function,momentum_function):
        self.force_function = force_function
        self.momentum_function = momentum_function

    def one_step(self,q_list,p_list,l_list,tau):

        p_list_1 = p_list + 0.5*tau*self.force_function.eval1(q_list,p_list,l_list,tau)

        q_list_1 = q_list + 0.5*tau*self.momentum_function.eval1(q_list,p_list_1,l_list,tau)
        q_list_1.retain_grad()
        q_list_1 = pbc(q_list_1,l_list)

        q_list_2 = q_list_1 + 0.5*tau*self.momentum_function.eval2(q_list_1,p_list_1,l_list,tau)
        q_list_2.retain_grad()
        q_list_2 = pbc(q_list_2,l_list)

        p_list_2 = p_list_1 + 0.5*tau*self.force_function.eval2(q_list_2,p_list_1,l_list,tau)

        assert_nan(p_list_2)
        assert_nan(q_list_2)
        return q_list_2,p_list_2,l_list

    def nsteps(self,q_list,p_list,l_list,tau,n_chain):
        for chain in range(n_chain):
            q_list,p_list,l_list = self.one_step(q_list,p_list,l_list,tau)
        return q_list,p_list,l_list
