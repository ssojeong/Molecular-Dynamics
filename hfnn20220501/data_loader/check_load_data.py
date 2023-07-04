import torch

from hamiltonian.lennard_jones2d import lennard_jones2d

class check_load_data:

    def __init__(self,qp_list_init, qp_list_final,l_list):

        self.q_list_init = qp_list_init[:,0,:,:]
        self.p_list_init = qp_list_init[:,1,:,:]
        self.q_list_final = qp_list_final[:,0,:,:]
        self.p_list_final = qp_list_final[:,1,:,:]
        self.l_list = l_list

        self.nsamples, self.nparticles, _ = self.q_list_init.shape
        self.potential_function = lennard_jones2d()

    def check(self):
        self.delta_total_energy()
        self.delta_momentum()

    def delta_total_energy(self):

        pe_init = self.potential_function.total_energy(self.q_list_init, self.l_list)
        pe_final = self.potential_function.total_energy(self.q_list_final, self.l_list)

        ke_init = torch.sum(self.p_list_init * self.p_list_init, dim=(1, 2)) * 0.5
        ke_final = torch.sum(self.p_list_final * self.p_list_final, dim=(1, 2)) * 0.5

        de = (ke_final + pe_final) - (ke_init + pe_init)

        max_de2 = torch.max(torch.abs(de))

        assert( max_de2 < 1e-4) , "error ... difference btw initial and final energy too big"
        print("difference btw initial and final energy not big ...",max_de2)

    def delta_momentum(self):

        pinit_sum = torch.sum(self.p_list_init, dim=1)  # shape [nsamples,dim]
        pfinal_sum = torch.sum(self.p_list_final, dim=1)  # shape [nsamples,dim]

        dp = pinit_sum - pfinal_sum  # shape [nsamples,dim]

        max_dp2 = torch.max(torch.abs(dp))

        assert( max_dp2 < 1e-4) , "error ... difference btw initial and final energy too big"
        print("difference btw initial and final momentum not big ...",max_dp2)
