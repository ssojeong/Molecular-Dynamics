from ML.LLUF.mb_base import mb_base
import torch

# ======================================================
# use this class when we use transformer only
# need to do reshape of input and output for transformer

class mb_transformer(mb_base):

    def __init__(self,mbnet_list,pwnet_list,ngrids,b,nnet,tau_init):
        super().__init__(mbnet_list,pwnet_list,ngrids,b,nnet,tau_init)
        print('--- initialize mb ff mlp ---')
        # SJ 20230701
        self.pwnet_list = pwnet_list
        self.mbnet_list = mbnet_list
        par = []
        for net in mbnet_list:
            par = par + list(net.parameters())
        for net in pwnet_list:
            par = par + list(net.parameters())
        self.param = par

    # ===================================================
    def evalall(self,net,x,dq): # do not use dq for mb_ff # SJ coord
        nsamples,nparticles,_,_,_ = self.mask.shape
        # x shape [nsamples * nparticles, ngrids * DIM * (q,p) * traj_len]
        y = net(x)
        dim = net.output_dim
        # reshape into [nsamples,nparticles,2]
        y3 = y.view([nsamples,nparticles,dim])
        return y3


