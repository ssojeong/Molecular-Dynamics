from ML.force_functions.mb_base import mb_base
import torch

# ======================================================

class mb_ff(mb_base):

    def __init__(self,net1,net2,ngrids,b,force_clip):
        super().__init__(net1,net2,ngrids,b)
        print('--- initialize mb ff ---')
        self.force_clip = force_clip

    # ===================================================
    def eval1(self,q_list,p_list,l_list,tau):
        force = self.evalall(self.net1,q_list,p_list,l_list,tau)
        force = torch.clamp(force,min=-self.force_clip,max=self.force_clip)
        return force
    # ===================================================
    def eval2(self,q_list,p_list,l_list,tau):
        force = self.evalall(self.net2,q_list,p_list,l_list,tau)
        force = torch.clamp(force,min=-self.force_clip,max=self.force_clip)
        return force


