import torch.nn as nn
import torch

class mockPWnet(nn.Module):

    # input is torch.cat(dq_sq, dp_sq)

    def __init__(self,input_dim,output_dim,nnodes,init_weights):
        super().__init__()
        assert output_dim==2,'make output dim =2'

    # x.shape = [batch,1]
    def forward(self,x):
        s1 = x*x + 3.2
        s2 = x*(x+1) - .2
        return torch.stack((s1,s2),dim=-1) # shape = [batch,2]
