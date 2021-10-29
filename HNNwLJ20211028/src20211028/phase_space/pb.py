import torch
from utils.get_paired_distance_indices import get_paired_distance_indices

class pb:

    ''' pb class is parent class that has phase_space child class.
        adjust particles in the boundary and calculate pair-wise distance btw two particles
    '''

    _obj_count = 0

    def __init__(self):

        pb._obj_count += 1
        assert(pb._obj_count <= 4), type(self).__name__ + ' has more than two objects'
        # one phase space object for the whole code
        # the other phase space object only use as a copy in lennard-jones class in dimensionless
        # form
        print('pb initialized')

    def adjust_real_q(self, q_list, l_list):
        ''' function to put particle back into boundary and use this function
        - class LJ terms, class hex_grids, class dpair_pbc(def paired_grid_xi)

        Parameters
        ----------
        q : torch.tensor
                shape is [nsamples, nparticle, DIM] par-par;
                [nsample, nparticle, 18, DIM=2]  par-grids;
        l_list : torch.tensor
                shape is [nsamples, nparticle, DIM] par-par ;
                [nsample, nparticle, 18, DIM=2]  par-grids ;

        Returns
        ----------
        adjust q in boundary condition
        shape is [nsamples, nparticle, DIM]
        '''

        indices = torch.where(torch.abs(q_list) > 0.5 * l_list)
        q_list[indices] = q_list[indices] - torch.round(q_list[indices] / l_list[indices]) * l_list[indices]
    # ===================================================

    def adjust_real_dq(self, dq_list, l_list):
        ''' function to put particle back into boundary and use this function in integrator method

        Parameters
        ----------
        dq_list : torch.tensor
                shape is [nsamples, nparticle, nparticle, DIM]
        l_list : torch.tensor
                shape is [nsamples, nparticle, DIM]

        Returns
        ----------
        adjust dq_list in boundary condition
        shape is [nsamples, nparticle, nparticle, DIM]
        '''
        llist0 = torch.unsqueeze(l_list, dim=1)
        llistm = torch.repeat_interleave(llist0, l_list.shape[1], dim=1)
        # lstatem.shape is [nsamples, nparticle, nparticle, DIM]

        indices = torch.where(torch.abs(dq_list) > 0.5 * llistm)

        dq_list[indices] = dq_list[indices] - torch.round(dq_list[indices] / llistm[indices]) * llistm[indices]


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
        # shape is [nsamples, nparticle, nparticle, DIM] [[q1, q2, q3, q4],[q1, q2, q3, q4],...,[q1, q2, q3, q4]]

        statet = statem.permute(get_paired_distance_indices.permute_order)
        # shape is [nsamples, nparticle, nparticle, DIM] [[q1, q1, q1, q1],[q2, q2, q2, q2],...,[q4, q4, q4, q4]]

        dstate = statet - statem
        # shape is [nsamples, nparticle, nparticle, DIM] [[q1-q1, q1-q2, q1-q3, q1-q4],...,[q4-q1, q4-q2, q4-q3, q4-q4]]

        return dstate
    # ===================================================
    def delta_state_pbc(self, state_list, l_list):
        ''' function to put particle back into boundary and use this function in integrator method

        Parameters
        ----------
        state_list : torch.tensor
                shape is [nsamples, nparticle, DIM]
        boxsize : torch.tensor
                shape is [nsamples, nparticle, DIM]
        '''
        dq = self.delta_state(state_list)
        self.adjust_real_dq(dq, l_list)

        return dq

    # ===================================================
    def paired_distance_reduced(self, q, nparticle, DIM):

        ''' function to calculate reduced distance btw two particles

        Parameters
        ----------
        q : torch.tensor
                shape is [nsamples, nparticle, DIM]
        qlen : nparticle
        q0 : new tensor with a dimension of size one inserted at the specified position (dim=1)
        qm : repeated tensor which has the same shape as q0 along with dim=1
        qt : permutes the order of the axes of a tensor
        dq : pair-wise distance btw two particles
        dq_reshape : obtain dq of non-zero indices
        dd : sum over DIM

        Returns
        ----------
        dq_reshape : pair-wise distances each DIM per nparticle
        dd : sum over DIM each particle
        '''

        l_list = torch.zeros(q.shape)
        l_list.fill_(1)

        dq = self.delta_state_pbc(q,l_list)

        dq_reduced_index = get_paired_distance_indices.get_indices(dq.shape)
        dq_flatten = get_paired_distance_indices.reduce(dq, dq_reduced_index)
        # dq_flatten.shape is [nsamples x nparticle x (nparticle - 1) x DIM]

        dq_reshape = dq_flatten.reshape((q.shape[0], nparticle, nparticle - 1, DIM))
        # dq_reshape.shape is [nsamples, nparticle, (nparticle - 1), DIM]

        dd = torch.sqrt(torch.sum(dq_reshape * dq_reshape, dim=-1))
        # dd.shape is [nsamples, nparticle, (nparticle - 1 )]

        return dq_reshape, dd

