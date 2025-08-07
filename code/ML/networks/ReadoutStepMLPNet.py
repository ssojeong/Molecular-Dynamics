import torch
import torch.nn as nn

class ReadoutStepMLPNet(nn.Module):

    def __init__(self,input_dim,output_dim,nnodes,p,readout=True):
        print('!!!!! update step mlp_net', input_dim, output_dim, nnodes, p)
        super().__init__()

        hidden_nodes = nnodes
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_nodes),
                                 nn.Dropout(p),
                                 nn.Linear(hidden_nodes, hidden_nodes),
                                 nn.Dropout(p),
                                 nn.Linear(hidden_nodes, output_dim),
                                 nn.Tanh()) if readout else nn.Identity()


    def forward(self,x):
        # x shape [nsamples * nparticles, embed_dim]
        x = self.mlp(x)
        # x shape [nsamples * nparticles, 2]
        return x

