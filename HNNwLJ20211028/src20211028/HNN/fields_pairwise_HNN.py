from hamiltonian.hamiltonian            import hamiltonian
from hamiltonian.lennard_jones          import lennard_jones
from hamiltonian.kinetic_energy         import kinetic_energy
from utils.get_paired_distance_indices  import get_paired_distance_indices
import itertools
import torch

class fields_pairwise_HNN(hamiltonian):

    def __init__(self, fhnn, pwhnn):

        super().__init__()

        self.fhnn = fhnn
        self.pwhnn = pwhnn

        super().append(lennard_jones())
        super().append(kinetic_energy())

    # ===================================================
    def get_netlist(self): # nochange
        return [self.fhnn.net1,self.fhnn.net2,self.pwhnn.net1,self.pwhnn.net2]
    # ===================================================
    def net_parameters(self):
        return itertools.chain(self.fhnn.net_parameters(),self.pwhnn.net_parameters())
    # ===================================================
    def dHdq1(self, phase_space):
        _, fforce = self.fhnn.dHdq1(phase_space)
        _, pwforce = self.pwhnn.dHdq1(phase_space)
        return _, fforce + pwforce
    # ===================================================
    def requires_grad_false(self):
        ''' use when predict trajectory as make requires_grad=False
         it means not update weights and bias'''

        for param in self.fhnn.net1.parameters():
            param.requires_grad = False

        for param in self.fhnn.net2.parameters():
            param.requires_grad = False

        for param in self.pwhnn.net1.parameters():
            param.requires_grad = False

        for param in self.pwhnn.net2.parameters():
            param.requires_grad = False
    # ===================================================
    def dHdq2(self, phase_space):
        _, fforce = self.fhnn.dHdq2(phase_space)
        _, pwforce = self.pwhnn.dHdq2(phase_space)
        return _, fforce + pwforce
    # ===================================================
    def train(self):
        '''pytorch network for training'''
        self.fhnn.net1.train()
        self.fhnn.net2.train()
        self.pwhnn.net1.train()
        self.pwhnn.net2.train()
    # ===================================================
    def eval(self):
        ''' pytorch network for eval '''
        self.fhnn.net1.eval()
        self.fhnn.net2.eval()
        self.pwhnn.net1.train()
        self.pwhnn.net2.train()
    # ===================================================
    def potential_rep(self, q_list, phase_space):
        ''' function to use for tune the function by adding an additional penalty term in loss'''

        boxsize = phase_space.get_l_list()
        nsamples, nparticle, DIM = q_list.shape
        # dq.shape = [nsamples, nparticle, DIM]

        q_list = q_list / boxsize #  dimensionless
        _, dr = phase_space.paired_distance_reduced(q_list, nparticle, DIM)

        s12 = 1 / pow(dr, 12)
        # s12.shape = [nsamples, nparticle, nparticle-1]

        boxsize = torch.mean(boxsize, dim=-1)
        # boxsize.shape is [nsamples, nparticle]
        boxsize = torch.unsqueeze(boxsize,dim=-1)
        # boxsize.shape is [nsamples, nparticle,1]

        term_dim = torch.sum(4 / pow(boxsize, 12) * s12, dim=-1)
        # term_dim.shape is [nsamples, nparticle]

        u_rep =  torch.sum(term_dim, dim=-1) * 0.5
        # term.shape is [nsamples]

        u_rep_max = torch.max(u_rep)

        return u_rep_max
    # ===================================================
    def delta_state(self, state_list):
        ''' function to calculate distance of q or p between two particles

        Parameters
        ----------
        state_list : torch.tensor
                shape is [nsamples, nparticle, DIM]
        statem : repeated tensor which has the same shape as state0 along with dim=1
        statet : permutes the order of the axes of a tensor

        Returns
        ----------
        dstate : distance of q or p btw two particles
        '''

        state_len = state_list.shape[1]  # nparticle
        state0 = torch.unsqueeze(state_list, dim=1)
        # shape is [nsamples, 1, nparticle, DIM]

        statem = torch.repeat_interleave(state0, state_len, dim=1)
        # shape is [nsamples, nparticle, nparticle, DIM]

        statet = statem.permute(get_paired_distance_indices.permute_order)
        # shape is [nsamples, nparticle, nparticle, DIM]

        dstate = statet - statem
        # shape is [nsamples, nparticle, nparticle, DIM]

        return dstate
