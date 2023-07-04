from ML.LLUF.pw_base import pw_base
import torch
import numpy as np
from utils.mydevice           import mydevice

class pw_null:

    def __init__(self,net_list,nnet,tau_init): # force_clip not used here
        self.nnet = nnet
        self.tau = self.tau()
        print('--- initialize pw_null ---')

    # ===================================================
    def parameters(self):
        return []
    # ===================================================
    def tau(self): # tensor
        # hk -- return torch.tensor(np.random.rand(self.nnet)*0,requires_grad=True,device = mydevice.get())
        return torch.zeros([self.nnet],requires_grad=True,device = mydevice.get()) #-- check shape hk
    # ===================================================
    def make_mask(self,nsamples,nparticles):
        # mask to mask out i-th particle interacted with itself
        pass
    # ===================================================
    def verbose(self,e,label):
        pass
    # ===================================================
    def prepare_q_input(self,pwnet_id,q_list,p_list,l_list):  # p_list not used here
        return 0
    # ===================================================
    def prepare_p_input(self,q_list,p_list,l_list): # q_list not used here
        return 0
    # ===================================================
    def grad_clip(self,clip_value):
        return 0
    # ===================================================
    def eval(self,net_id,q_input_list,p_input_list):
        return 0
    # ===================================================
    def train_mode(self):
        pass
    # ===================================================
    def eval_mode(self):
        pass
