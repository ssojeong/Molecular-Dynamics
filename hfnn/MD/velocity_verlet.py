from utils.pbc import pbc
from utils.utils import assert_nan

class velocity_verlet:

    def __init__(self,force_function):
        self.force_function = force_function

    def one_step(self,q_list,p_list,l_list,tau):
        f = self.force_function.eval1(q_list,p_list,l_list,tau)
        p_list_2 = p_list + 0.5*tau*self.force_function.eval1(q_list,p_list,l_list,tau)
        q_list_2 = q_list + tau*p_list_2
        q_list_2.retain_grad()
        q_list_new = pbc(q_list_2,l_list)
        p_list_new = p_list_2 + 0.5*tau*self.force_function.eval2(q_list_new,p_list_2,l_list,tau)
        assert_nan(p_list_new)
        assert_nan(q_list_new)
        return q_list_new,p_list_new,l_list

    def nsteps(self,q_list,p_list,l_list,tau,n_chain):
        for chain in range(n_chain):
            q_list,p_list,l_list = self.one_step(q_list,p_list,l_list,tau)
        return q_list,p_list,l_list

