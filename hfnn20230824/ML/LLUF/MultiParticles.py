import torch
import torch.nn as nn

class MultiParticles(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    # take in output of single_particle
    # and output to update_step
    # x.shape=[nsample,nparticle,embed_dim]
    # q_previous.shape=[nsample,nparticle,dim]
    # output shape=[nsample,nparticle,embed_dim]
    def eval(self, x, q_previous):
        nsample, nparticle, embed_dim = x.shape
        x = x.reshape(nsample * nparticle, embed_dim)
        y = self.net(x, q_previous)
        return y.reshape(nsample, nparticle, -1)

