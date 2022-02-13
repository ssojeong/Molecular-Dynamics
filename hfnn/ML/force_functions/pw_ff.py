from ML.force_functions.pw_base import pw_base
import torch

class pw_ff(pw_base):

    def __init__(self,net1,net2,force_clip): # force_clip not used here
        super().__init__(net1,net2)
        print('--- initialize pw_ff ---')
        self.force_clip = force_clip

    def eval1(self,q_list,p_list,l_list,tau):
        force = self.evalall(self.net1,q_list,p_list,l_list,tau)
        force = torch.clamp(force,min=-self.force_clip,max=self.force_clip)
        return force

    def eval2(self,q_list,p_list,l_list,tau):
        force = self.evalall(self.net2,q_list,p_list,l_list,tau)
        force = torch.clamp(force,min=-self.force_clip,max=self.force_clip)
        return force


