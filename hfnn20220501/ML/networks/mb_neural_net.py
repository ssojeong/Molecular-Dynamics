import torch.nn as nn
import torch
from ML.networks.base_net import base_net

class mb_neural_net(base_net):

    def __init__(self,input_dim,output_dim,nnodes):
        super().__init__()

        hidden_nodes = nnodes
        h1 = hidden_nodes
        h2 = hidden_nodes
        h3 = hidden_nodes
 
        fc1 = nn.Linear(input_dim,h1,bias=True)
        fc2 = nn.Linear(h1,h2,bias=True)
        fc3 = nn.Linear(h2,h3,bias=True)
        fc4 = nn.Linear(h3,output_dim,bias=True)

        self.output_dim = output_dim
        self.layers = nn.ModuleList([fc1,fc2,fc3,fc4])
        self.layers.apply(self.init_weights)

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

    def forward(self,x):
        for m in self.layers:
            x = m(x)
            x = self.activation(x)
        return x

