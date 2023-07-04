import torch.nn as nn
import torch


class neural_net3(nn.Module):

    def __init__(self,input_dim,output_dim):
        super(neural_net3,self).__init__()

        h1 = 64 #256 # 1024 # 64
        h2 = 64 #256 # 1024 # 64
        self.fc1 = nn.Linear(input_dim,h1,bias=True)
        #self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1,h2,bias=True)
        #self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2,output_dim,bias=True)

        self.output_dim = output_dim
        self.weight_clamp_value = 100

        self.layers = nn.ModuleList([self.fc1,self.fc2,self.fc3])
        self.layers.apply(self.init_weights)
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

        print('weight/bias range [',maxw,minw,']')

    def activation(self,x):
        return torch.tanh(x)

    def forward(self,x):
        x = self.fc1(x)
        #x = self.bn1(x)
        x = self.activation(x)
        x = self.fc2(x)
        #x = self.bn2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x

    def clamp_weights(self):
        self.layers.apply(self.tanh_clamp)


