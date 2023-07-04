import torch.nn as nn
import torch
from ML.networks.base_net import base_net

class pw_neural_net(base_net):

    # input is torch.cat(dq_sq, dp_sq)

    def __init__(self,input_dim,output_dim,nnodes):
        super().__init__()

        hidden_nodes = nnodes
        h1 = hidden_nodes
        h2 = hidden_nodes//2
        h3 = hidden_nodes
        h4 = output_dim
        fc1 = nn.Linear(input_dim,h1,bias=True)
        fc2 = nn.Linear(h1,h2,bias=True)
        fc3 = nn.Linear(h2,h3,bias=True)
        fc4 = nn.Linear(h3,h4,bias=True)

        self.output_dim = output_dim

        self.layers = nn.ModuleList([fc1,fc2,fc3,fc4])
        self.layers.apply(self.init_weights)

 
    def init_weights(self,m): # m is layer that is nn.Linear
        if type(m) == nn.Linear:
            # set the xavier_gain neither too much bigger than 1, nor too much less than 1
            # recommended gain value for the given nonlinearity function
            # tanh gain=5/3
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))  
            m.bias.data.fill_(0.0)

    def activation(self,x):
        return torch.tanh(x)

    #def make_rotational_invariant(self,x):
    #    x2 = x*x 
    #    dq2 = torch.unsqueeze(x2[:,0]+x2[:,1],1) # dqx*dqx+dqy*dqy
    #    dp2 = torch.unsqueeze(x2[:,2]+x2[:,3],1) # dpx*dpx+dpy*dpy
    #    tau = torch.unsqueeze(x[:,4],1)
    #    new_x = torch.cat((dq2,dp2,tau),dim=1)
    #    return new_x

    def forward(self,x):

        #x = self.make_rotational_invariant(x)
        for m in self.layers[:-1]:
            x = m(x)
            x = self.activation(x)
        return self.layers[-1](x)

