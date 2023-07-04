from utils.pbc import delta_state
from utils.pbc import delta_pbc
from utils.mydevice import mydevice
from ML.force_functions.fbase import fbase
import torch
import itertools
import torch.nn as nn

class pw_base(fbase):

    def __init__(self,net_list,nnet):
        super().__init__(net_list,nnet)

        self.net_list = net_list
        par = []
        for net in net_list:
            par = par + list(net.parameters())
        self.param = par
        self.mask = None

        print('pw fnn')

    # ===================================================
    def make_mask(self,nsamples,nparticles):
        dim = self.net_list[0].output_dim
        # mask to mask out self force when doing net predictions
        self.mask = torch.ones([nsamples,nparticles,nparticles,dim],device=mydevice.get())
        dia = torch.diagonal(self.mask,dim1=1,dim2=2)
        dia.fill_(0.0)
        #chek = torch.ones([nsamples,nparticles,nparticles,dim],device=mydevice.get())
        #for np in range(nparticles):
        #    chek[:,np,np,:] = 0.0
        #assert (torch.all(torch.eq(chek,self.mask))),'diagonal method failed'
    # ===================================================
    def grad_clip(self,clip_value):
        for net in self.net_list:
            nn.utils.clip_grad_value_(net.parameters(),clip_value)
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
        dq0 = delta_pbc(q_list, l_list)
        dq1 = self.make_correct_shape(dq0)
        return dq1
    # ===================================================
    def prepare_p_input(self,q_list,p_list,l_list): # q_list not used here
        dp0 = delta_state(p_list)
        dp1 = self.make_correct_shape(dp0)
        return dp1
    # ===================================================
    # return network trainable parameters including tau
    def parameters(self): 
        return self.param                 
    # ===================================================
    def eval(self,q_input_list,p_input_list):
        netid, y = self.eval_base(q_input_list,p_input_list)
        return self.tau[netid]*y
    # ===================================================
    def evalall(self,net,x,dq):
        y = net(x,dq)
        y2 = self.unpack_dqdp(y,net.output_dim)
        #y3 = torch.clamp(y2,min=-clip_value,max=clip_value)
        return y2
    # ===================================================
    def unpack_dqdp(self,y,dim):

        nsamples,nparticles,_,dim = self.mask.shape
        y1 = y.view(nsamples, nparticles, nparticles, dim)
        y2 = y1*self.mask

        y3 = torch.sum(y2, dim=2) # y2.shape = [nsamples,nparticles,2]
        return y3

