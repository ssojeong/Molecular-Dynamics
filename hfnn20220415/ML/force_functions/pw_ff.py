from ML.force_functions.pw_base import pw_base
import torch
import itertools

class pw_ff(pw_base):

    def __init__(self,net_list,neval=6): # force_clip not used here
        super().__init__(net_list,neval)
        print('--- initialize pw_ff ---')

