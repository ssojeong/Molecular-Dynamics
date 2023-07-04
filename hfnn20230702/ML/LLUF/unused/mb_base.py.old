import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchviz import make_dot

#from inspect import currentframe

#from utils.utils     import assert_nan
from utils.mydevice  import mydevice
from ML.force_functions.fbase import fbase
from utils.pbc import pbc
from utils.pbc import delta_pbc
from utils.get_paired_distance_indices import get_paired_distance_indices

# ======================================================
def paired_distance_reduced(q, npar):

    l_list = torch.zeros(q.shape)
    l_list.fill_(1)
    l_list = mydevice.load(l_list)

    dq = delta_pbc(q,l_list) # shape is [nsamples, nparticle, nparticle, DIM]

    dq_reduced_index = get_paired_distance_indices.get_indices(dq.shape)
    dq_flatten = get_paired_distance_indices.reduce(dq, dq_reduced_index)
    # dq_flatten.shape is [nsamples x nparticle x (nparticle - 1) x DIM]

    dq_reshape = dq_flatten.view(q.shape[0], npar, npar - 1, q.shape[2])
    # dq_reshape.shape is [nsamples, nparticle, (nparticle - 1), DIM]

    dd = torch.sqrt(torch.sum(dq_reshape * dq_reshape, dim=-1))
    # dd.shape is [nsamples, nparticle, (nparticle - 1 )]
    return dq_reshape, dd
# ======================================================

class mb_base(fbase):

    def __init__(self,net1,net2,ngrids,b):
        super().__init__(net1,net2)
        self.ngrids = ngrids
        self.b = b
        print('mb fnn')

    # ===================================================
    def evalall(self,net,q_list,p_list,l_list,tau):
        # prepare input to feed into network
        x = self.prepare_input(q_list,p_list,l_list,tau,self.ngrids,self.b)
        y = net(x)
        # y1.shape = [nsamples,nparticle,2] -- for force
        # y1.shape = [nsamples,nparticle,1] -- for hamiltonian
        dim = net.output_dim
        y_shape = torch.Size((q_list.shape[0],q_list.shape[1],dim))
        y = y.view(y_shape)
        return y
    # ===================================================
    def prepare_input(self,q_list,p_list,l_list,tau,ngrids,b): # make dqdp for n particles

        nsamples, nparticle, DIM = q_list.shape

        u = self.make_grids_center(q_list,l_list,b) # position at grids
        # u.shape is [nsample, nparticle*ngrids, DIM=2]
        u_fields = self.gen_u_fields(q_list, l_list, u,ngrids)
        # u_fields.shape is [nsamples, npartice, grids*DIM]

        v_fields = self.gen_v_fields(q_list,p_list,u, l_list,ngrids)
        tau_tensor = torch.zeros([nsamples, nparticle, 1])
        tau_tensor.fill_(tau * 0.5)
        tau_tensor = mydevice.load(tau_tensor)

        x = torch.cat((u_fields, v_fields, tau_tensor), dim=-1)
        # x.shape = shape is [ nsamples, nparticle,  ngrids * DIM + ngrids * DIM + 1]

        x = x.view(nsamples * nparticle, 2 * ngrids * DIM + 1)
        return x
    # ===================================================
    def hex_grids_list(self,b):
        grids_ncenter = torch.tensor([[-b * 0.5, -b], [-b * 0.5, b], [-b, 0.], [b, 0.], [b * 0.5, -b], [b * 0.5, b]])
        # grids_ncenter.shape is [6, 2]
        grids_ncenter = mydevice.load(grids_ncenter)
        return grids_ncenter
    # ===================================================
    def make_grids_center(self,q, l_list,b):
        '''make_grids function to shift 6 grids points at (0,0) to each particle position as center'''
        l_list = torch.unsqueeze(l_list, dim=2)
        # boxsize.shape is [nsamples, nparticle, 1, DIM]
        l_list = l_list.repeat_interleave(self.hex_grids_list(b).shape[0], dim=2)

        q_list = torch.unsqueeze(q, dim=2)
        # q_list.shape is [nsamples, nparticle, 1, DIM=(x coord, y coord)]

        grids_ncenter = self.hex_grids_list(b) + q_list
        # grids_ncenter.shape is [6, 2] + [nsamples, nparticle, 1, DIM] => [nsample, nparticle, 6, DIM=2]

        pbc(grids_ncenter, l_list)  # pbc - for grids
        #self.show_grids_nparticles(q, grids_ncenter,l_list[0,0,0])

        grids_ncenter = grids_ncenter.view(-1, q.shape[1] * self.hex_grids_list(b).shape[0], q.shape[2])
        # shape is [nsamples, nparticle*ngrids, DIM=(x,y)]
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
    def gen_u_fields(self, q, l_list, u_list, ngrids, _dphi_maxcut=108.35):
        nsamples, nparticle, DIM = q.shape
        _u_fields = self.ufields(q, l_list, u_list, ngrids)
        # shape is [ nsamples, nparticle*ngrids, DIM ]

        mask1 = _u_fields > _dphi_maxcut
        if mask1.any() == True:
            _u_fields[mask1] = _dphi_maxcut

        mask2 = _u_fields < - _dphi_maxcut # HK, why got negative maxcut?
        if mask2.any() == True:
            _u_fields[mask2] = - _dphi_maxcut

        #assert_nan(_u_fields,currentframe())

        _gen_u_fields = _u_fields.view(nsamples, nparticle, -1)
        # shape is [ nsamples, nparticle, ngrids*DIM ]
        return _gen_u_fields
    # ===================================================
    def ufields(self,q, l_list, u_list, ngrids):
        # u_list.shape is [nsample, nparticle*ngrids, DIM=2]
        nsamples, npar, DIM = q.shape
        xi = q / l_list  # dimensionless
        # l_list.shape is [nsamples, nparticle, DIM]

        l_list4u = l_list.repeat_interleave(ngrids, dim=1)
        u_dimless = u_list / l_list4u  # dimensionless

        l_reduced = torch.unsqueeze(l_list, dim=2)  # l_list.shape is [nsamples, nparticle, 1, DIM]
        l_reduced = l_reduced.repeat_interleave(u_list.shape[1], dim=2)
        # l_reduced.shape is [nsamples, nparticle, nparticle * ngrids, DIM]
        l_reduced.fill_(1)

        delta_grid_xi, d = self.dpair_pbc(xi, u_dimless, l_reduced)
        # delta_grid_xi.shape is [nsamples, nparticle, nparticle * ngrids, DIM]
        # d.shape is [nsamples, nparticle, nparticle * ngrids]

        l_list = l_list[:, :, 0]
        # l_list.shape is [nsamples, nparticle]
        l_list = l_list.view(nsamples, npar, 1, 1)
        # l_list.shape is [nsamples, nparticle, 1 ,1 ]

        eps = 1e-10

        a12 = (4 * 1 * pow(1, 12)) / (pow(l_list, 13)+eps)
        a6 = (4 * 1 * pow(1, 6)) / (pow(l_list, 7)+eps)

        d = torch.unsqueeze(d, dim=-1)

        # d.shape is [nsamples, nparticle, nparticle * ngrids, 1]

        s12_ = -12 * (delta_grid_xi) / (pow(d, 14)+eps)
        s6_ = -6 * (delta_grid_xi) / (pow(d, 8)+eps)

        # shape is [nsamples, nparticle, nparticle * ngrids, DIM]

        s12 = self.zero_ufields(s12_)
        s6 = self.zero_ufields(s6_)

        dphi_fields = torch.sum(a12 * s12, dim=1) - torch.sum(a6 * s6, dim=1)  # np.sum axis=1 j != k
        # dphidxi.shape is [nsamples, nparticle * ngrids, DIM=2]

        return dphi_fields
    # ===================================================
    def zero_ufields(self,s12s6):

        nsamples, npar, npar_ngrids, DIM = s12s6.shape
        make_zero_ufields = s12s6.view(nsamples, npar, npar, npar_ngrids // npar, DIM)
        # make_zero_phi.shape is [nsamples, nparticle, nparticle, ngrids, DIM]

        dy = torch.diagonal(make_zero_ufields, 0, 1, 2)  # offset, nparticle, nparticle
        torch.fill_(dy, 0.0)

        s12s6_reshape = make_zero_ufields.view(nsamples, npar, npar_ngrids, DIM)
        # s12s6_reshape.shape is [nsamples, nparticle, nparticle*ngrids, DIM]
        return s12s6_reshape
    # ===================================================
    def dpair_pbc(self,q, u_list, l_list):  #

        # all list dimensionless
        q_state = torch.unsqueeze(q, dim=2)
        # shape is [nsamples, nparticle, 1, DIM=(x,y)]
        u_list = torch.unsqueeze(u_list, dim=1)

        # shape is [nsamples, 1, nparticle* ngrids, DIM=(x,y)]
        paired_grid_q = u_list - q_state

        # paired_grid_q.shape is [nsamples, nparticle, nparticle * ngrids, DIM]
        pbc(paired_grid_q, l_list)

        dd = torch.sqrt(torch.sum(paired_grid_q * paired_grid_q, dim=-1))

        # dd.shape is [nsamples, nparticle, nparticle * ngrids]
        return paired_grid_q, dd
    # ===================================================
    def gen_v_fields(self,q,p,u_list,l_list,ngrids):

        l_list = torch.unsqueeze(l_list, dim=2)
        l_list = l_list.repeat_interleave(u_list.shape[1], dim=2)
        # boxsize.shape is [nsamples, nparticle, nparticle * ngrids, DIM]

        nsamples, nparticle, DIM = p.shape

        _, d = self.dpair_pbc(q, u_list, l_list)
        # d.shape is [nsamples, nparticle, nparticle * ngrids]

        # r^2 nearest distance weight
        weights = 1 / (d*d + 1e-10)  # update!
        # weights.shape is [nsamples, nparticle, nparticle * ngrids]

        weights = torch.unsqueeze(weights, dim=-1)
        # w_thrsh.shape is [nsamples, nparticle, nparticle * ngrids, 1]

        p_list = torch.unsqueeze(p, dim=2)
        # p_list.shape is [nsamples, nparticle, 1, DIM]

        wp = weights * p_list
        # wp.shape [nsamples, nparticle, nparticle * ngrids, DIM]

        wp_nume = torch.sum(wp, dim=1)
        # wp_nume.shape [nsamples,  nparticle * ngrids, DIM]
        wp_deno = torch.sum(weights, dim=1)
        # wp_deno.shape is [nsamples, nparticle * ngrids, 1]

        p_ngrids = wp_nume / wp_deno
        # p_grids.shape [nsamples,  nparticle * ngrids, DIM]

        p_ngrids = p_ngrids.view(nsamples, nparticle, ngrids, DIM)
        # p_ngrids.shape [nsamples, npartice, grids, DIM]
        # p_list.shape is [nsamples, nparticle, 1, DIM]

        # relative momentum : center of particles
        relative_p = p_ngrids - p_list
        # relative_p.shape [nsamples, npartice, grids, DIM]

        relative_p = relative_p.view(nsamples, nparticle, ngrids * DIM)
        # relative_p.shape [nsamples, npartice, grids*DIM]
        return relative_p


# =====================================================================
# ============================================
# ===== test codes form here =================
# ============================================
def check_dphi(q, l_list):

    nsamples, nparticle, DIM = q.shape
    xi = q / l_list # dimensionless

    l_list = l_list[:,:,0]
    l_list = l_list.view(nsamples, nparticle,1,1)

    delta_xi, d = paired_distance_reduced(xi, nparticle)
    # delta_xi.shape is [nsamples, nparticle, (nparticle - 1), DIM]
    # d.shape is [nsamples, nparticle, (nparticle - 1 )]

    d = torch.unsqueeze(d, dim=-1)
    # d.shape is [nsamples, nparticle, (nparticle - 1 ), 1]

    a12 = (4 * 1 * pow(1, 12)) / pow(l_list, 13)
    a6 = (4 * 1 * pow(1, 6)) / pow(l_list, 7)

    s12 = -12 * (delta_xi) / pow(d, 14)
    s6 = -6 * (delta_xi) / pow(d, 8)
    # shape is [nsamples, nparticle, (nparticle - 1 ), DIM]

    dphidxi = torch.sum( a12*s12, dim=2) - torch.sum( a6*s6, dim=2)  # np.sum axis=2 j != k ( nsamples-1)
    # dphidxi.shape is [nsamples, nparticle, DIM]

    return dphidxi

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
def get_forces(ff, q, p, l, tau):
    return ff.eval1(q,p,l,tau), ff.eval2(q,p,l,tau)

# ============================================
def read_data(nsamples,npar,dim):

    # l_list = torch.rand([nsamples,1,dim])*10
    # l_list = torch.repeat_interleave(l_list,npar,dim=1)

    l_list = torch.rand([nsamples])*10
    l_list= l_list.reshape(nsamples,1,1)
    l_list = torch.repeat_interleave(l_list,npar,dim=1)
    l_list = torch.repeat_interleave(l_list,dim,dim=2)

    q = l_list*(torch.rand([nsamples,npar,dim],requires_grad=True)-0.5)
    p = torch.rand([nsamples,npar,dim],requires_grad=True)

    return q,p,l_list

# ============================================
if __name__=='__main__':

    #torch.manual_seed(222314) # too close  two particles
    torch.manual_seed(214)

    nsamples = 88
    npar = 2
    dim = 2
    ngrids = 6
    b = 0.001
    tau = 0.1

    net1 = test_neural_net(2*ngrids*dim+1,2)
    net2 = test_neural_net(2*ngrids*dim+1,2)
    ff = mb_force_function(net1,net2,ngrids,b)

    for e in range(2000):
        qc,pc,lc = read_data(nsamples,npar,dim)
        
        qs = torch.split(qc,nsamples//2)
        ps = torch.split(pc,nsamples//2)
        ls = torch.split(lc,nsamples//2)
        
        q0 = qs[0]
        q1 = qs[1]
        p0 = ps[0]
        p1 = ps[1]
        l0 = ls[0]
        l1 = ls[1]
        
        fcq,fcp = get_forces(ff,qc,pc,lc,tau)
        f0q,f0p = get_forces(ff,q0,p0,l0,tau)
        f1q,f1p = get_forces(ff,q1,p1,l1,tau)
        
        
        fcatq = torch.cat((f0q,f1q),0)
        fcatp = torch.cat((f0p,f1p),0)
        diffq = (fcq-fcatq)
        diffp = (fcp-fcatp)
        sum_diffq = torch.sum(diffq*diffq) 
        sum_diffp = torch.sum(diffp*diffp) 
        
        if sum_diffq>1e-7 or sum_diffp>1e-7:
            print(e,'mse btw forces ',sum_diffq,' ',sum_diffp)

    quit()

    #u_list = ff.make_grids_center(q,l_list,b)
    #u_fields = ff.gen_u_fields(q, l_list, u_list, ngrids)
    #dphi = check_dphi(q,l_list)
    #print(u_fields)
    #print(dphi)

    #p = p + 0.5*tau*ff.eval1(q,p,l_list,tau)
    #q = tau*p
    #p = p + 0.5*tau*ff.eval2(q,p,l_list,tau)
  
    #print_compute_tree('mb_ptree',p)
    #print_compute_tree('mb_qtree',q)


