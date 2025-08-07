import torch
import torch.nn as nn

class SingleParticle(nn.Module):

    # two networks first one for updating p
    # second one for updating q
    # networks are all mb-net, no more pw-net
    def __init__(self, net):
        super().__init__()
        self.net = net

    # take in prepared data and output feature for
    # multi-particle layers
    # input x.shape [nsample, nparticle, traj_len, ngrid * DIM * (q,p)]
    # output.shape = [nsample, nparticle, embed_dim]
    def eval(self,x):
        nsample,nparticle,traj_len,_ = x.shape
        x = x.reshape(nsample*nparticle,traj_len,-1)
        output = self.net(x)
        return output.reshape(nsample, nparticle, -1)

