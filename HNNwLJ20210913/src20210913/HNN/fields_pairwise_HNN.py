from HNN.HNN_base                       import HNN_base
from utils.get_paired_distance_indices  import get_paired_distance_indices
import itertools
import torch

class fields_pairwise_HNN:

    def __init__(self, fhnn, pwhnn):

        self.fhnn = fhnn
        self.pwhnn = pwhnn

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
    def potential_rep(self, q_list):
        ''' function to use for tune the function by adding an additional penalty term in loss'''

        dq = self.delta_state(q_list)
        # dq.shape = [nsamples, nparticle, nparticle, 2]

        dr2 = torch.sum(dq*dq, dim=-1)
        # dr2.shape = [nsamples, nparticle, nparticle]

        dr2min = torch.min(dr2[dr2.nonzero(as_tuple=True)])

        u_rep = 4 * dr2min**(-6)

        return u_rep
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
