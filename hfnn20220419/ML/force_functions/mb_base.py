import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchviz import make_dot
import itertools

#from inspect import currentframe

from ML.force_functions.fbase import fbase
from utils.mydevice  import mydevice
from utils.pbc import pbc
from utils.pbc import delta_pbc
from utils.pbc import delta_state
from utils.get_paired_distance_indices import get_paired_distance_indices

# ======================================================

class mb_base(fbase):

    def __init__(self,mbnet_list,pwnet_list,ngrids,b,nnet):
        super().__init__(mbnet_list,nnet)

        #print('mb_base: number of net made, mb_net ',len(mbnet_list),' pwnet ',len(pwnet_list))
        #print('nnet ',nnet)
        self.ngrids = ngrids
        self.b = b
        self.pwnet_list = pwnet_list
        self.mbnet_list = mbnet_list
        par = []
        for net in mbnet_list:
            par = par + list(net.parameters())
        for net in pwnet_list:
            par = par + list(net.parameters())
        self.param = par
        self.mask = None
        print('mb fnn')

    # ===================================================
    def make_mask(self,nsamples,nparticles):
        dim = self.mbnet_list[0].output_dim
        # mask to mask out self force when doing net predictions
        self.mask = torch.ones([nsamples,nparticles,nparticles,self.ngrids,dim],device=mydevice.get())
        dia = torch.diagonal(self.mask,dim1=1,dim2=2)
        dia.fill_(0.0)
        #chek = torch.ones([nsamples,nparticles,nparticles,self.ngrids,dim],device=mydevice.get())
        #for np in range(nparticles):
        #    chek[:,np,np,:,:] = 0.0
        #assert (torch.all(torch.eq(chek,self.mask))),'diagonal method failed'
    # ===================================================
    # return network trainable parameters including tau
    def parameters(self): 
        return self.param
    # ===================================================
    def eval(self,q_input_list,p_input_list):
        netid, y = self.eval_base(q_input_list,p_input_list)
        return self.tau[netid]*y
    # ===================================================
    def evalall(self,net,x,dq): # do not use dq for mb_ff
        nsamples,nparticles,_,_,_ = self.mask.shape
        y = net(x)
        dim = net.output_dim
        # reshape into [nsamples,nparticles,2]
        y3 = y.view([nsamples,nparticles,dim])
        return y3
    # ===================================================
    def grad_clip(self,clip_value):
        for net in self.mbnet_list:
            nn.utils.clip_grad_value_(net.parameters(),clip_value)
        for net in self.pwnet_list:
            nn.utils.clip_grad_value_(net.parameters(),clip_value)
    # ===================================================
    def prepare_q_input(self,pwnet_id,q_list,p_list,l_list): # make dqdp for n particles
        return self.prepare_q_input_net(q_list,p_list,l_list,self.pwnet_list[pwnet_id])
    # ===================================================
    def prepare_q_input_net(self,q_list,p_list,l_list,pwnet): # make dqdp for n particles

        self.nsamples, self.nparticles, DIM = q_list.shape

        u = self.make_grids_center(q_list,l_list,self.b) # position at grids

        # u.shape is [nsamples, nparticles*ngrids, DIM=2]
        u_fields = self.gen_u_fields(q_list,p_list, l_list,u,pwnet,self.ngrids)  # force fields
        # u_fields.shape  [nsamples, npartice, grids*DIM]

        u_fields = u_fields.view(self.nsamples*self.nparticles,self.ngrids * DIM) 
        # [batch, 12] -- good

        return u_fields # 
    # ===================================================
    def prepare_p_input(self,q_list,p_list,l_list): # make dqdp for n particles

        nsamples, nparticles, DIM = p_list.shape

        u = self.make_grids_center(q_list,l_list,self.b) # position at grids

        v_fields = self.gen_v_fields(q_list,p_list,u,l_list,self.ngrids) # velocity fields - no change
        # v_fields.shape [nsamples, npartice, grids*DIM]

        # x = torch.cat((u_fields, v_fields), dim=-1)
        # x.shape = shape is [ nsamples, nparticles,  ngrids * DIM + ngrids * DIM ]

        v_fields = v_fields.view(nsamples * nparticles, self.ngrids * DIM) # [batch, 12] -- good

        return v_fields # 
    # ===================================================
    def hex_grids_list(self,b):
        grids_ncenter = torch.tensor([[-b * 0.5, -b], [-b * 0.5, b], [-b, 0.], [b, 0.], [b * 0.5, -b], [b * 0.5, b]])
        # grids_ncenter.shape is [6, 2]
        grids_ncenter = mydevice.load(grids_ncenter)
        return grids_ncenter
    # ===================================================
    def make_grids_center(self,q,l_list,b):
        '''make_grids function to shift 6 grids points at (0,0) to each particle position as center'''

        l_list = torch.unsqueeze(l_list, dim=2)
        # l_list.shape is [nsamples, nparticles, 1, DIM]

        l_list = l_list.repeat_interleave(self.hex_grids_list(b).shape[0], dim=2)

        q_list = torch.unsqueeze(q, dim=2)
        # q_list.shape is [nsamples, nparticles, 1, DIM=(x coord, y coord)]

        grids_ncenter = self.hex_grids_list(b) + q_list
        # grids_ncenter.shape is [6, 2] + [nsamples, nparticles, 1, DIM] => [nsamples, nparticles, 6, DIM=2]

        pbc(grids_ncenter, l_list)  # pbc - for grids
        #self.show_grids_nparticles(q, grids_ncenter,l_list[0,0,0])

        grids_ncenter = grids_ncenter.view(-1, q.shape[1] * self.hex_grids_list(b).shape[0], q.shape[2])
        # shape is [nsamples, nparticles*ngrids, DIM=(x,y)]
        return grids_ncenter
    # ===================================================
    def show_grids_nparticles(self,q_list, u_list, boxsize):

        bs = boxsize.detach().numpy()

        for i in range(1):  # show two samples

            plt.title('sample {}'.format(i))
            plt.xlim(-bs[0] / 2, bs[0] / 2)
            plt.ylim(-bs[1] / 2, bs[1] / 2)
            plt.plot(u_list[i, :, :, 0].detach().numpy(), u_list[i, :, :, 1].detach().numpy(), marker='.', color='k',
                     linestyle='none', markersize=12)
            plt.plot(q_list[i, :, 0].detach().numpy(), q_list[i, :, 1].detach().numpy(), marker='x', color='r',
                     linestyle='none', markersize=12)
            plt.show()
            plt.close()
    # ===================================================
    def gen_u_fields(self, q, p, l_list, u_list, pwnet, ngrids, _dphi_maxcut=500): #108.35):

        nsamples, nparticles, DIM = q.shape
        _u_fields = self.ufields(q, p, l_list, u_list,pwnet, ngrids)
        # shape is [ nsamples, nparticles*ngrids, DIM ]

        # use neural net to predice force, no need maxcut
        mask1 = _u_fields > _dphi_maxcut
        if mask1.any() == True:
        #    _u_fields[mask1] = _dphi_maxcut
            print('force predicition for mb grid too high')
            print('max force ',torch.max(_u_fields))
            quit()

        mask2 = _u_fields < - _dphi_maxcut # HK, why got negative maxcut?
        if mask2.any() == True:
        #    _u_fields[mask2] = - _dphi_maxcut
            print('force predicition for mb grid too high')
            print('max force ',torch.min(_u_fields))
            quit()

        #assert_nan(_u_fields,currentframe())

        _gen_u_fields = _u_fields.view(nsamples, nparticles, -1)
        # shape is [ nsamples, nparticles, ngrids*DIM ]
        return _gen_u_fields
    # ===================================================
    def ufields(self,q, p, l_list, u_list, pwnet, ngrids):  # u_list = grid center position
        # u_list.shape is [nsamples, nparticles*ngrids, DIM=2]
        # l_list.shape is [nsamples, nparticles, DIM]
        nsamples, nparticles, DIM = q.shape
        xi = q / l_list  # dimensionless

        l_list4u = l_list.repeat_interleave(ngrids, dim=1)
        u_dimless = u_list / l_list4u  # dimensionless

        l_reduced = torch.ones(l_list.shape,requires_grad=False,device=mydevice.get()) # shape [nsamples,nparticles,dim]
        l_reduced = torch.unsqueeze(l_reduced, dim=2)              # shape is [nsamples, nparticles, 1, DIM]
        l_reduced = l_reduced.repeat_interleave(u_list.shape[1], dim=2)
        # l_reduced.shape is [nsamples, nparticles, nparticles * ngrids, DIM]

        _, d_sq = self.dpair_pbc_sq(xi, u_dimless, l_reduced)
        # d.shape is [nsamples, nparticles, nparticles * ngrids]

        l_list = l_list[:, :, 0]
        l_list = l_list.view(nsamples, nparticles, 1)
        # l_list.shape = [nsamples, nparticles, 1]
        dq_sq = d_sq * l_list * l_list

        dq_sq = dq_sq.view(nsamples*nparticles*nparticles*ngrids,1) # dq^2
        # dq.shape is [nsamples * nparticles * nparticles * ngrids, 1]

        del_p = delta_state(p) #  shape is [nsamples, nparticles, nparticles, DIM]

        dp_sq = torch.sum(del_p * del_p, dim=-1) #  shape is [nsamples, nparticles, nparticles]

        dp_sq = torch.unsqueeze(dp_sq, dim=3)  # l_list.shape is [nsamples, nparticles, nparticles,1]
        dp_sq = dp_sq.repeat_interleave(ngrids, dim=3)
        # dp.shape is [nsamples, nparticles, nparticles, ngrids ]

        dp_sq = dp_sq.view(nsamples * nparticles * nparticles * ngrids, 1)  # |dp|
        # dp.shape is [nsamples * nparticles * nparticles * ngrids, 1]

        x = torch.cat((dq_sq, dp_sq), dim=-1)
        # x.shape = [batch, 2] - dq^2, dp^2
        pair_pwnet = pwnet(x,dq_sq)
        # pair_pwnet.shape = [batch, 2] - fx, fy

        pair_pwnet = self.zero_ufields(pair_pwnet, nsamples, nparticles, ngrids)
        # pair_pwnet.shape is [nsamples, nparticles, nparticles*ngrids, DIM]

        dphi_fields = torch.sum(pair_pwnet, dim=1)  # np.sum axis=2 j != k ( nsamples-1)
        # dphi_fields.shape is [nsamples, nparticles * ngrids, DIM=2]
        return dphi_fields
    # ===================================================
    def zero_ufields(self,pair_pwnet,nsamples, nparticles, ngrids):
        # pair_pwnet shape is, [nsamples * nparticles * nparticles * ngrids, 2]

        _, DIM = pair_pwnet.shape
        pair_pwnet1 = pair_pwnet.view(nsamples, nparticles, nparticles, ngrids, DIM)
        # make_zero_phi.shape is [nsamples, nparticles, nparticles, ngrids, DIM]

        pair_pwnet2 = pair_pwnet1*self.mask

        pair_pwnet3 = pair_pwnet2.view(nsamples, nparticles, nparticles * ngrids, DIM)
        # shape is [nsamples, nparticles, nparticles*ngrids, DIM]

        return pair_pwnet3
    # ===================================================
    def dpair_pbc_sq(self,q, u_list, l_list):  #

        # all list dimensionless
        q_state = torch.unsqueeze(q, dim=2)
        # shape is [nsamples, nparticles, 1, DIM=(x,y)]
        u_list = torch.unsqueeze(u_list, dim=1)

        # shape is [nsamples, 1, nparticles* ngrids, DIM=(x,y)]
        paired_grid_q = u_list - q_state

        # paired_grid_q.shape is [nsamples, nparticles, nparticles * ngrids, DIM]
        pbc(paired_grid_q, l_list)

        dd = torch.sum(paired_grid_q * paired_grid_q, dim=-1)

        # dd.shape is [nsamples, nparticles, nparticles * ngrids]
        return paired_grid_q, dd
    # ===================================================
    def gen_v_fields(self,q,p,u_list,l_list,ngrids): # velocity fields

        l_list = torch.unsqueeze(l_list, dim=2)
        l_list = l_list.repeat_interleave(u_list.shape[1], dim=2)
        # boxsize.shape is [nsamples, nparticles, nparticles * ngrids, DIM]

        nsamples, nparticles, DIM = p.shape

        _, d_sq = self.dpair_pbc_sq(q, u_list, l_list)
        # d.shape is [nsamples, nparticles, nparticles * ngrids]

        # r^2 nearest distance weight
        weights = 1 / (d_sq + 1e-10)  # update!
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

        relative_p = relative_p.view(nsamples, nparticles, ngrids * DIM)
        # relative_p.shape [nsamples, npartice, grids*DIM]
        return relative_p


# =====================================================================

# ============================================
class test_neural_net(nn.Module):

    def __init__(self,input_dim,output_dim):
        super().__init__()

        self.fc = nn.Linear(input_dim,output_dim,bias=False)

    def forward(self,x):
        return self.fc(x)

# ============================================
def print_compute_tree(name,node):
    dot = make_dot(node)
    #print(dot)
    dot.render(name)
# ============================================
def get_forces(ff, q, p, l):
    return ff.eval1(q,p,l), ff.eval2(q,p,l)

# ============================================
def read_data(nsamples,nparticles,dim):

    # l_list = torch.rand([nsamples,1,dim])*10
    # l_list = torch.repeat_interleave(l_list,nparticles,dim=1)

    l_list = torch.rand([nsamples])*10
    l_list= l_list.reshape(nsamples,1,1)
    l_list = torch.repeat_interleave(l_list,nparticles,dim=1)
    l_list = torch.repeat_interleave(l_list,dim,dim=2)

    q = l_list*(torch.rand([nsamples,nparticles,dim],requires_grad=True)-0.5)
    p = torch.rand([nsamples,nparticles,dim],requires_grad=True)

    return q,p,l_list

# ============================================


