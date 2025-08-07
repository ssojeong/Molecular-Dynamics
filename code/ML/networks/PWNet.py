import torch.nn as nn
import torch

class PWNet(nn.Module):

    # input is torch.cat(dq_sq, dp_sq)

    def __init__(self,input_dim,output_dim,nnodes,init_weights):
        super().__init__()

        hidden_nodes = nnodes
        h1 = hidden_nodes
        #h1 = max(hidden_nodes,input_dim)
        h2 = hidden_nodes
        h3 = hidden_nodes
        h4 = hidden_nodes
        h5 = output_dim
        fc1 = nn.Linear(input_dim,h1,bias=True)
        fc2 = nn.Linear(h1,h2,bias=True)
        fc3 = nn.Linear(h2,h3,bias=True)
        fc4 = nn.Linear(h3,h4,bias=True)
        fc5 = nn.Linear(h4,h5,bias=True)

        self.output_dim = output_dim

        self.layers = nn.ModuleList([fc1,fc2,fc3,fc4,fc5])

        if init_weights == 'tanh':
            self.layers.apply(self.init_weights_tanh)
        else:
            self.layers.apply(self.init_weights_relu)

        self.inv_max_force = 1./10.0
        #self.inv_max_force = 1./2.0
        self.inv_max_expon = 3
        

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

    def factor(self,dq):
        return 1.0/( dq**self.inv_max_expon + self.inv_max_force )

    def forward(self,x):
        dq=x
        # pwnet input: x shape [nsamples*nparticles*nparticles, 2*traj_len]
        # pwnet input for mbnet input : x shape [nsamples * nparticles * nparticles * ngrids, 2]
        for m in self.layers:
            x = m(x)
            if m != self.layers[-1]:
                x = self.relu(x)
            else:
                x = self.tanh(x) #;print('pw layer', m, x)
        w = self.factor(dq)
        # pwnet output: x shape [nsamples*nparticles*nparticles, 2]
        # pwnet output for mbnet input : x shape [nsamples * nparticles * nparticles * ngrids, 2]
        return x*w

