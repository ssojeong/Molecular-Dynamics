from ML.LLUF.pw_base import pw_base
import torch
import itertools

class pw_mlp(pw_base):

    def __init__(self,net_list,nnet,tau_init): # force_clip not used here
        super().__init__(net_list,nnet,tau_init)
        print('--- initialize pw_ff ---')

