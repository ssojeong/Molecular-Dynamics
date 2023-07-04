import torch.nn as nn
import torch

class mb_neural_net(nn.Module):

    def __init__(self,input_dim,output_dim):
        super(mb_neural_net,self).__init__()

        h1 = 256
        h2 = 256
        h3 = 256
        h4 = 256
 
        fc1 = nn.Linear(input_dim,h1,bias=True)
        fc2 = nn.Linear(h1,h2,bias=True)
        fc3 = nn.Linear(h2,h3,bias=True)
        fc4 = nn.Linear(h3,h4,bias=True)
        fc5 = nn.Linear(h4,output_dim,bias=True)

        self.output_dim = output_dim
        self.layers = nn.ModuleList([fc1,fc2,fc3,fc4,fc5])
        self.layers.apply(self.init_weights)
        self.weight_clamp_value = 100
        self.clamp_weights()

    def init_weights(self,m): # m is layer that is nn.Linear
        if type(m) == nn.Linear:
            # set the xavier_gain neither too much bigger than 1, nor too much less than 1
            # recommended gain value for the given nonlinearity function
            # tanh gain=5/3
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))  
            m.bias.data.fill_(0.0)

    def tanh_clamp(self,m):
        wc = self.weight_clamp_value
        if type(m) == nn.Linear:
            m.weight.data = torch.tanh(m.weight.data/wc)*wc
            m.bias.data   = torch.tanh(m.bias.data/wc  )*wc

    def weight_range(self):
        
        maxw = -1e10
        minw =  1e10
        for m in self.layers:
            if type(m) == nn.Linear:
                maxw = max(maxw,torch.max(m.weight).item())
                maxw = max(maxw,torch.max(m.bias).item())
                minw = min(minw,torch.min(m.weight).item())
                minw = min(minw,torch.min(m.bias).item())
        print('weight/bias range [',minw,maxw,']')
        #print('weight/bias range not used ')

    def clamp_weights(self):
        self.layers.apply(self.tanh_clamp)

    def activation(self,x):
        return torch.tanh(x)

    def forward(self,x):
        for m in self.layers[:-1]:
            x = m(x)
            x = self.activation(x)
        return self.layers[-1](x)

