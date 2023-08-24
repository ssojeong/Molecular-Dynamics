import torch
import torch.nn as nn
from utils.mydevice import mydevice

from utils.pbc import pbc
from utils.pbc import _delta_state
import matplotlib.pyplot as plt

class PrepareData(nn.Module):

    def __init__(self, net, ngrids, b):
        super().__init__()
        # net : mb4pw -- use to extract features for position variable of grid point
        self.net = net
        self.ngrids = ngrids
        self.b = b

    def cat_qp(self,q_input_list,p_input_list):
        # # q_input_list : mb net [phi0, phi1, phi2, ...] # phi0=(phi0x,phi0y)
        # # phi shape [nsamples, nparticles, ngrids * DIM]
        # # p_input_list : mb net [pi0, pi1, pi2, ...] # pi0=(pi0x,pi0y)
        # # return mbnet qp_cat phi, pi along time
        q_input_list = torch.stack(q_input_list,dim=2)
        p_input_list = torch.stack(p_input_list,dim=2)
        # # phi shape [nsamples, nparticles, traj_len, ngrids * DIM]
        qp_cat = torch.cat((q_input_list,p_input_list), dim=-1)
        # shape [nsamples, nparticles, traj_len, ngrids * DIM * (q,p)]
        return qp_cat

    # ===================================================
    def make_mask(self,nsamples,nparticles,dim):
        # mask to mask out only hexagonal grids centered at i-th particle
        # and then use to make zero of fields which interact with itself
        self.mask = torch.ones([nsamples,nparticles,nparticles,self.ngrids,dim],device=mydevice.get())
        dia = torch.diagonal(self.mask,dim1=1,dim2=2) # [nsamples, ngrids, dim, nparticles]
        dia.fill_(0.0)

    # ===================================================
    def prepare_q_feature_input(self, q_list, l_list):  # make dqdp for n particles
        self.nsamples, self.nparticles, self.DIM = q_list.shape
        uli_list = self.make_grids_center(q_list, l_list, self.b)  # position at grid points
        # uli_list.shape is [nsamples, nparticles * ngrids, DIM=2]
        q_var = self.gen_qvar(q_list, l_list, uli_list, self.net, self.ngrids)  # force fields
        # u_fields.shape  [nsamples, npartice, ngrids * DIM], ngrids=6, DIM=2
        return q_var

    # ===================================================
    def prepare_p_feature_input(self, q_list, p_list, l_list):  # make dqdp for n particles
        uli_list = self.make_grids_center(q_list, l_list, self.b)  # position at grids
        # shape is [nsamples, nparticles * ngrids, DIM=(x,y)]
        p_var = self.gen_pvar(q_list, p_list, uli_list, l_list, self.ngrids)  # velocity fields - no change
        # v_fields.shape [nsamples, npartice, grids * DIM]
        return p_var

    # ===================================================
    def hex_grids_list(self, b):
        grids_ncenter = torch.tensor([[-b * 0.5, -b], [-b * 0.5, b], [-b, 0.], [b, 0.], [b * 0.5, -b], [b * 0.5, b]])
        # grids_ncenter.shape is [6, 2]
        grids_ncenter = mydevice.load(grids_ncenter)
        return grids_ncenter

    # ===================================================
    def make_grids_center(self, q, l_list, b):
        '''make_grids function to shift 6 grids points at (0,0) to each particle position as center'''

        l_list = torch.unsqueeze(l_list, dim=2)
        # l_list.shape is [nsamples, nparticles, 1, DIM]

        l_list = l_list.repeat_interleave(self.hex_grids_list(b).shape[0], dim=2)

        q_list = torch.unsqueeze(q, dim=2)
        # q_list.shape is [nsamples, nparticles, 1, DIM=(x coord, y coord)]

        grids_ncenter = self.hex_grids_list(b) + q_list
        # grids_ncenter.shape is [6, 2] + [nsamples, nparticles, 1, DIM] => [nsamples, nparticles, 6, DIM=2]

        pbc(grids_ncenter, l_list)  # pbc - for grids
        # self.show_grids_nparticles(q, grids_ncenter,l_list[0,0,0])

        grids_ncenter = grids_ncenter.view(-1, q.shape[1] * self.hex_grids_list(b).shape[0], q.shape[2])
        # shape is [nsamples, nparticles * ngrids, DIM=(x,y)]
        return grids_ncenter

    # ===================================================
    def dpair_pbc_sq(self, q, uli_list, l_list):  #

        # all list dimensionless
        q_state = torch.unsqueeze(q, dim=2)
        # shape is [nsamples, nparticles, 1, DIM=(x,y)]
        uli_list = torch.unsqueeze(uli_list, dim=1)
        # shape is [nsamples, 1, nparticles * ngrids, DIM=(x,y)]
        paired_grid_q = uli_list - q_state
        # paired_grid_q.shape is [nsamples, nparticles, nparticles * ngrids, DIM]
        pbc(paired_grid_q, l_list)

        dd = torch.sum(paired_grid_q * paired_grid_q, dim=-1)
        # dd.shape is [nsamples, nparticles, nparticles * ngrids]
        return paired_grid_q, dd

    # ===================================================
    def gen_qvar(self, q_list, l_list, uli_list, pwnet, ngrids, _dphi_maxcut=200):  # 108.35):
        # uli_list.shape is [nsamples, nparticles * ngrids, DIM=2]
        nsamples, nparticles, DIM = q_list.shape
        _qvar = self.qvar(q_list, l_list, uli_list, pwnet, ngrids)
        # shape is [ nsamples, nparticles * ngrids, DIM ]

        # use neural net to predice force, no need maxcut
        mask1 = _qvar > _dphi_maxcut
        if mask1.any() == True:
            #    _u_fields[mask1] = _dphi_maxcut
            print('force predicition for mb grid too high')
            print('max force ', torch.max(_qvar))
            quit()

        mask2 = _qvar < - _dphi_maxcut  # HK, why got negative maxcut?
        if mask2.any() == True:
            #    _u_fields[mask2] = - _dphi_maxcut
            print('force predicition for mb grid too high')
            print('max force ', torch.min(_qvar))
            quit()

        # assert_nan(_u_fields,currentframe())

        _gen_qvar = _qvar.view(nsamples, nparticles, -1)
        # shape is [ nsamples, nparticles, ngrids * DIM ]

        return _gen_qvar
    # ===================================================
    def qvar(self, q_list, l_list, uli_list, pwnet, ngrids):

        nsamples, nparticles, DIM = q_list.shape

        _, pair_pwnet = self.pairwise_qvar(q_list, l_list, uli_list, pwnet, ngrids)
        pair_pwnet = pair_pwnet.view(nsamples, nparticles, nparticles * ngrids, DIM)
        # shape is [nsamples, nparticles, nparticles * ngrids, DIM]

        #pair_pwnet = self.zero_qvar(pair_pwnet, nsamples, nparticles, ngrids)
        ## pair_pwnet.shape is [nsamples, nparticles, nparticles*ngrids, DIM]

        q_var = torch.sum(pair_pwnet, dim=1)  # np.sum axis=2 j != k ( nsamples-1)
        # q_var.shape is [nsamples, nparticles * ngrids, DIM=2]
        return q_var

    # ===================================================
    def pairwise_qvar(self, q_list, l_list, uli_list, pwnet, ngrids):  # uli_list = grid center position
        # uli_list.shape is [nsamples, nparticles * ngrids, DIM=2]
        # l_list.shape is [nsamples, nparticles, DIM]
        nsamples, nparticles, DIM = q_list.shape

        # for computation convenienc and numerical stability rescale the system so that the boxsize 1
        q_dimless = q_list / l_list  # dimensionless

        l_list4uli = l_list.repeat_interleave(ngrids, dim=1)
        uli_dimless = uli_list / l_list4uli  # dimensionless

        l_reduced = torch.ones(l_list.shape, requires_grad=False,device=mydevice.get())
        # shape [nsamples,nparticles,dim]
        l_reduced = torch.unsqueeze(l_reduced, dim=2)
        # shape is [nsamples, nparticles, 1, DIM]
        l_reduced = l_reduced.repeat_interleave(uli_list.shape[1], dim=2)
        # l_reduced.shape is [nsamples, nparticles, nparticles * ngrids, DIM]

        _, d_sq = self.dpair_pbc_sq(q_dimless, uli_dimless, l_reduced)
        # d_sq.shape is [nsamples, nparticles, nparticles * ngrids]

        l_list = l_list[:, :, 0]
        l_list = l_list.view(nsamples, nparticles, 1)
        # l_list.shape = [nsamples, nparticles, 1]
        dq_sq = d_sq * l_list * l_list

        dq_sq = dq_sq.view(nsamples * nparticles * nparticles * ngrids, 1)  # dq^2
        # dq_sq.shape is [nsamples * nparticles * nparticles * ngrids, 1]

        pair_pwnet = pwnet(dq_sq)
        # pair_pwnet.shape = [batch, 2]

        return dq_sq, pair_pwnet

    # ===================================================
    def gen_pvar(self, q, p, uli_list, l_list, ngrids):  # velocity fields

        l_list = torch.unsqueeze(l_list, dim=2)
        l_list = l_list.repeat_interleave(uli_list.shape[1], dim=2)
        # boxsize.shape is [nsamples, nparticles, nparticles * ngrids, DIM]

        nsamples, nparticles, DIM = p.shape

        _, d_sq = self.dpair_pbc_sq(q, uli_list, l_list)
        # d_sq.shape is [nsamples, nparticles, nparticles * ngrids]

        # r^2 nearest distance weight
        weights = 1 / (d_sq + 1e-10)
        # weights.shape is [nsamples, nparticles, nparticles * ngrids]

        weights = torch.unsqueeze(weights, dim=-1)
        # w_thrsh.shape is [nsamples, nparticles, nparticles * ngrids, 1]

        p_list = torch.unsqueeze(p, dim=2)
        # p_list.shape is [nsamples, nparticles, 1, DIM]

        wp = weights * p_list
        # wp.shape [nsamples, nparticles, nparticles * ngrids, DIM]

        wp_nume = torch.sum(wp, dim=1)
        # wp_nume.shape [nsamples,  nparticles * ngrids, DIM]
        wp_deno = torch.sum(weights, dim=1)
        # wp_deno.shape is [nsamples, nparticles * ngrids, 1]

        p_ngrids = wp_nume / wp_deno
        # p_grids.shape [nsamples,  nparticles * ngrids, DIM]

        p_ngrids = p_ngrids.view(nsamples, nparticles, ngrids, DIM)
        # p_ngrids.shape [nsamples, npartice, grids, DIM]
        # p_list.shape is [nsamples, nparticles, 1, DIM]

        # relative momentum : center of particles
        relative_p = p_ngrids - p_list
        # relative_p.shape [nsamples, npartice, grids, DIM]

        gen_pvar = relative_p.view(nsamples, nparticles, ngrids * DIM)
        # relative_p.shape [nsamples, npartice, grids*DIM]
        return gen_pvar


    # ===================================================
    def zero_qvar(self, pair_pwnet, nsamples, nparticles, ngrids):
        # pair_pwnet shape is, [nsamples * nparticles * nparticles * ngrids, 2]

        _, DIM = pair_pwnet.shape
        pair_pwnet1 = pair_pwnet.view(nsamples, nparticles, nparticles, ngrids, DIM)
        # make_zero_phi.shape is [nsamples, nparticles, nparticles, ngrids, DIM]

        pair_pwnet2 = pair_pwnet1 * self.mask

        pair_pwnet3 = pair_pwnet2.view(nsamples, nparticles, nparticles * ngrids, DIM)
        # shape is [nsamples, nparticles, nparticles*ngrids, DIM]

        return pair_pwnet3

    # ===================================================
    def show_grids_nparticles(self, q_list, uli_list, boxsize):

        bs = boxsize.detach().numpy()

        for i in range(1):  # show two samples

            plt.title('sample {}'.format(i))
            plt.xlim(-bs[0] / 2, bs[0] / 2)
            plt.ylim(-bs[1] / 2, bs[1] / 2)
            plt.plot(uli_list[i, :, :, 0].detach().numpy(), uli_list[i, :, :, 1].detach().numpy(), marker='.', color='k',
                     linestyle='none', markersize=12)
            plt.plot(q_list[i, :, 0].detach().numpy(), q_list[i, :, 1].detach().numpy(), marker='x', color='r',
                     linestyle='none', markersize=12)
            plt.show()
            plt.close()

