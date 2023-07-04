import torch
import torch.nn as nn
import numpy as np

from utils.mydevice import mydevice

class fbase:

    def __init__(self,nnet,t_init):
    #def __init__(self,net_list,nnet,t_init): # 20230701 hk
        #self.net_list = net_list # 20230701 hk
        tau_init = np.random.rand(nnet)*t_init # change form 0.01 to 0.001
        self.tau = torch.tensor(tau_init,requires_grad=True,device = mydevice.get())

    def cat_qp(self,q_input_list,p_input_list):
        # # q_input_list : mb net [phi0, phi1, phi2, ...], pw net [dq0, dq1, dq2, ...] # phi0=(phi0x,phi0y)
        # # p_input_list : mb net [pi0, pi1, pi2, ...], pw net [dp0, dp1, dp2, ...] # pi0=(pi0x,pi0y)
        # # return mbnet qp_cat phi, pi along time
        # # return pwnet qp_cat dq, dp along time
        qp_list = [item for sublist in zip(q_input_list, p_input_list) for item in sublist]
        # mbnet qp_list [phi0, pi0, phi1, pi1, phi2, pi2, ...]
        # # phi shape [nsamples * nparticles, ngrids * DIM]
        # # dq shape [nsamples * nparticles * nparticles, 2]
        qp_cat = torch.cat(qp_list, dim=-1) # shape [nsamples * nparticles, ngrids * DIM * (q,p) * traj_len]
        return qp_cat

    # ===================================================
    def verbose(self,e,label):
        tau_list = self.tau.tolist()
        valu_label = label + ' value'
        print(e,label,' '.join('{}: {:.2e}'.format(*k) for k in enumerate(tau_list)))

        if self.tau.grad is not None:
            tau_grad_list = self.tau.grad.tolist()
            grad_label = label + ' grad'
            print(e,grad_label,' '.join('{}:{:.2e}'.format(*k) for k in enumerate(tau_grad_list)))
        else:
            print('tau grad is None before gradient')

        self.f_stat.print(e,label)
        self.f_stat.clear()

    # ===================================================
    def clip_tau_grad(self,clip_value):
        nn.utils.clip_grad_value_(self.tau,clip_value)

    # ===================================================
    def get_max_abs_grad(self,name,net_list): # SJ update

        grads = []
        for idx,net in enumerate(net_list):
            #print(name,'net idx ------ ',idx,' -----------')
            for p in net.parameters():
                #print(name,' value ',p)
                #print(name,' grad  ',p.grad)
                grads.append(p.grad.view(-1))
        grads = torch.abs(torch.cat(grads))
        return max(grads).item()

    # ===================================================
    def set_requires_grad_false(self,net_list): 
        for idx,net in enumerate(net_list):
            net.eval()
            for p in net.parameters():
                p.requires_grad = False
        self.tau.requires_grad = False
    # ===================================================
    def set_requires_grad_true(self,net_list): 
        for idx,net in enumerate(net_list):
            net.train()
            for p in net.parameters():
                p.requires_grad = True
        self.tau.requires_grad = True
    # ===================================================


