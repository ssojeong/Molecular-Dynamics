import torch
import torch.nn as nn
from torch.autograd import grad

from ML.LLUF.pw_base import pw_base

# ======================================================
class pw_hf(pw_base):

    def __init__(self,net1,net2,force_clip):
        super().__init__(net1,net2)
        print('--- initialize pw_hf ---')
        self.force_clip = force_clip
    # ===================================================
    def eval1(self,q_list,p_list,l_list,tau):
        H = self.evalall(self.net1,q_list,p_list,l_list,tau)
        # do autograd here
        self.zero_grad(q_list)
        force = -grad(H,q_list,create_graph=True, grad_outputs=torch.ones_like(H))[0]
        force = torch.clamp(force,min=-self.force_clip,max=self.force_clip)
        return force
    # ===================================================
    def eval2(self,q_list,p_list,l_list,tau):
        H = self.evalall(self.net2,q_list,p_list,l_list,tau)
        # do autograd here
        self.zero_grad(q_list)
        force = -grad(H,q_list,create_graph=True, grad_outputs=torch.ones_like(H))[0]
        force = torch.clamp(force,min=-self.force_clip,max=self.force_clip)
        return force
    # ===================================================
    def zero_grad(self,q_list):
        if q_list.grad is not None: q_list.grad.data.zero_()


