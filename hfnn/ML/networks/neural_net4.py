import torch.nn as nn
import torch

class neural_net4(nn.Module):

    def __init__(self,input_dim,output_dim):
        super(neural_net4,self).__init__()

        h1 = 256
        h2 = 256
        h3 = 256
 
        self.fc1 = nn.Linear(input_dim,h1,bias=True)
        self.fc2 = nn.Linear(h1,h2,bias=True)
        self.fc3 = nn.Linear(h2,h3,bias=True)
        self.fc4 = nn.Linear(h3,output_dim,bias=True)

        # initialize weights
        self.fc1.weight.data.fill_(0.1/h1)
        self.fc1.bias.data.fill_(0.0)
        self.fc2.weight.data.fill_(0.1/h2)
        self.fc2.bias.data.fill_(0.0)
        self.fc3.weight.data.fill_(0.1/h3)
        self.fc3.bias.data.fill_(0.0)
        self.fc4.weight.data.fill_(0.01/output_dim)
        self.fc4.bias.data.fill_(0.0)

    def activation(self,x):
        return torch.tanh(x)

    def forward(self,x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        return x


