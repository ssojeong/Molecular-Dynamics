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

        #self.inv_max_force = 1./20.0
        self.inv_max_force = 1./2.0
        self.inv_max_expon = 3
        
 
    def init_weights(self,m): # m is layer that is nn.Linear
        if type(m) == nn.Linear:
            # set the xavier_gain neither too much bigger than 1, nor too much less than 1
            # recommended gain value for the given nonlinearity function
            # tanh gain=5/3
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))  
            #m.weight.data = m.weight.data*1e-1
            m.bias.data.fill_(0.0)

    def activation(self,x):
        return torch.tanh(x)

    def factor(self,dq):
        return 1.0/( dq**self.inv_max_expon + self.inv_max_force )

    def forward(self,x,dq):

        for m in self.layers:
            x = m(x)
            x = self.activation(x)
        w = self.factor(dq)
        return x*w

