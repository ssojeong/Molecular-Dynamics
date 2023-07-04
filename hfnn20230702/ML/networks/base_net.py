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


