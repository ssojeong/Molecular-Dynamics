import torch.nn as nn
import torch
from ML.networks.base_net import base_net

class mb_mlp_net(base_net):

    def __init__(self,input_dim,output_dim,nnodes,h_mode,p):
        super().__init__()

        print('!!!!! mb_mlp_net', input_dim, output_dim, nnodes, h_mode, p)

        hidden_nodes = nnodes
        h1 = hidden_nodes
        #h1 = max(hidden_nodes,input_dim)
        h2 = hidden_nodes
        h3 = hidden_nodes
        h4 = hidden_nodes
 
        fc1 = nn.Linear(input_dim,h1,bias=True)
        fc2 = nn.Linear(h1,h2,bias=True)
        fc3 = nn.Linear(h2,h3,bias=True)
        fc4 = nn.Linear(h3,h4,bias=True)
        fc5 = nn.Linear(h4,output_dim,bias=True)

        self.p = p
        self.output_dim = output_dim
        self.layers = nn.ModuleList([fc1,fc2,fc3,fc4,fc5])

        if h_mode == 'tanh':
            self.layers.apply(self.init_weights_tanh)
        else:
            self.layers.apply(self.init_weights_relu)

    def init_weights_tanh(self,m): # m is layer that is nn.Linear
        if type(m) == nn.Linear:
            # set the xavier_gain neither too much bigger than 1, nor too much less than 1
            # recommended gain value for the given nonlinearity function
            # tanh gain=5/3
            #print('init weight; tanh......')
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
            #m.weight.data = m.weight.data*1e-1
            m.bias.data.fill_(0.0)

    def init_weights_relu(self,m): # m is layer that is nn.Linear
        if type(m) == nn.Linear:
            # set the xavier_gain neither too much bigger than 1, nor too much less than 1
            # recommended gain value for the given nonlinearity function
            # tanh gain=5/3
            #print('init weight; relu.......')
            nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
            m.bias.data.fill_(0.0)

    def relu(self,x):
        return torch.relu(x)

    def tanh(self,x):
        return torch.tanh(x)

    def dropout(self,x,p):
        return nn.functional.dropout(x,p, training=True)

    def forward(self,x):
        # x shape [nsamples * nparticles, ngrids * (q,p) * DIM * traj_len]
        for m in self.layers:
            x = m(x)
            if m != self.layers[-1]:
              x = self.relu(x)
              x = self.dropout(x,p=self.p) 
              #print('output layer dp', m, x)
            else: x = self.tanh(x) #;print('mb layer', m, x)
        # x shape [nsamples * nparticles, 2]
        return x

