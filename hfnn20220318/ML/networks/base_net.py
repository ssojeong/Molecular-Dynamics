import torch.nn as nn
import torch


class base_net(nn.Module):

    def __init__(self): 
        super().__init__()
        pass

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

    #def clamp_weights(self):
        #print('clamp weights defunct')
        #self.layers.apply(self.tanh_clamp)



    #def tanh_clamp(self,m):
    #    wc = self.weight_clamp_value
    #    if type(m) == nn.Linear:
    #        m.weight.data = torch.tanh(m.weight.data/wc)*wc
    #        m.bias.data   = torch.tanh(m.bias.data/wc  )*wc


