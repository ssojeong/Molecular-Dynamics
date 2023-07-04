from ML.force_functions.mb_base import mb_base
import torch

# ======================================================

class mb_ff(mb_base):

    def __init__(self,mbnet_list,pwnet_list,ngrids,b,nnet):
        super().__init__(mbnet_list,pwnet_list,ngrids,b,nnet)
        print('--- initialize mb ff ---')



