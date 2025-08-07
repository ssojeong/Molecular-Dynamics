import torch.nn as nn

class ReadoutStep(nn.Module):

    def __init__(self, net):
        super().__init__()
        self.net = net

    # take in output of single_particle
    # and output to update_step
    # x.shape=[nsample,nparticle,embed_dim]
    # output shape=[nsample,nparticle,dim]
    def eval(self, x):
        nsample,nparticle,embed_dim = x.shape
        x = x.reshape(nsample*nparticle,embed_dim)
        y = self.net(x)
        return y.reshape(nsample,nparticle,-1)
