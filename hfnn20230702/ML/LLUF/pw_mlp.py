from utils.force_stat         import force_stat
from utils.pbc                import _delta_state
from utils.pbc                import pairwise_dq_pbc
from utils.mydevice           import mydevice
from ML.LLUF.fbase import fbase
import torch
import itertools
import torch.nn as nn

class pw_mlp(fbase):

    def __init__(self,net_list,nnet,tau_init):
        #super().__init__(net_list,nnet,tau_init) # 20230701 hk
        super().__init__(nnet,tau_init)

        self.net_list = net_list
        par = []
        for net in net_list:
            par = par + list(net.parameters())
        self.param = par
        self.mask = None
        self.f_stat = force_stat(nnet)

        print('pw fnn')

    # ===================================================
    def max_abs_grad(self,name):
        pw_name = name + '-pw4pw'
        return self.get_max_abs_grad(pw_name,self.net_list) # SJ update
    # ===================================================
    def train_mode(self): 
        self.set_requires_grad_true(self.net_list)
    # ===================================================
    def eval_mode(self): 
        self.set_requires_grad_false(self.net_list)
    # ===================================================
    def make_mask(self,nsamples,nparticles):
        # mask to mask out i-th particle interacted with itself
        dim = self.net_list[0].output_dim
        self.mask = torch.ones([nsamples,nparticles,nparticles,dim],device=mydevice.get())
        dia = torch.diagonal(self.mask,dim1=1,dim2=2) # [nsamples, dim, nparticle]
        dia.fill_(0.0)
    # ===================================================
    def grad_clip(self,clip_value):
        for net in self.net_list:
            nn.utils.clip_grad_value_(net.parameters(),clip_value)
        self.clip_tau_grad(clip_value) # HK
    # ===================================================
    def make_correct_shape(self,qp):

        nsamples,nparticles,_,_ = self.mask.shape
        qp = torch.sum(qp*qp,dim=-1)

        qp = torch.unsqueeze(qp, dim=3)  # [nsamples,nparticles,nparticles,1]
        qp = qp.view(nsamples * nparticles * nparticles, 1)
        # shape is [nsamples*nparticles*nparticles, 1]
        return qp
    # ===================================================
    def prepare_q_input(self,pwnet_id,q_list,p_list,l_list):  # p_list not used here
        dq0 = pairwise_dq_pbc(q_list, l_list)
        dq1 = self.make_correct_shape(dq0)
        return dq1
    # ===================================================
    def prepare_p_input(self,q_list,p_list,l_list): # q_list not used here
        dp0 = _delta_state(p_list)
        dp1 = self.make_correct_shape(dp0)
        return dp1
    # ===================================================
    # return network trainable parameters including tau
    def parameters(self): 
        return self.param                 
    # ===================================================
    def eval(self,net_id,q_input_list,p_input_list):
        # q_input_list [dq0, dq1, dq2, ...]
        # p_input_list [dp0, dp1, dp2, ...]
        y = self.eval_base(net_id,q_input_list,p_input_list) #fbase
        f = torch.abs(self.tau[net_id])*y
        self.f_stat.accumulate(net_id,y)
        return f
    # ===================================================
    def eval_base(self,net_id,q_input_list,p_input_list):
        x = self.cat_qp(q_input_list,p_input_list)
        # mbnet x shape [nsamples * nparticles, ngrids * DIM * (q,p) * traj_len]
        # pwnet x shape [nsamples * nparticles * nparticles, (q,p) * traj_len]
        dq = torch.mean(torch.stack(q_input_list),dim=0)
        target_net = self.net_list[net_id]
        return self.evalall(target_net,x,dq)
    # ===================================================
    def evalall(self,net,x,dq):
        # x shape [nsamples*nparticles*nparticles, 2*traj_len]
        y = net(x,dq)
        # y shape [nsamples*nparticles*nparticles, 2]
        y2 = self.unpack_dqdp(y,net.output_dim)
        return y2
    # ===================================================
    def unpack_dqdp(self,y,dim):

        nsamples,nparticles,_,dim = self.mask.shape
        y1 = y.view(nsamples, nparticles, nparticles, dim)
        y2 = y1*self.mask

        y3 = torch.sum(y2, dim=2) # y2.shape = [nsamples,nparticles,2]
        return y3

