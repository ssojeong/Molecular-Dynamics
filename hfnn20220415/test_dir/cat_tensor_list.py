import torch
import torch.nn as nn
import itertools
import torch.optim as optim

x = torch.rand([3,4])

a = None
b = None
y = torch.reshape(x, (a, b))

'''
class net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,1)


net1 = net()
net2 = net()
tau = torch.ones([2])

net_list = [net1,net2]

net_param = []
for i in net_list:
    net_param = net_param + list(i.parameters())

param = net_param + [tau]

print('param ',param)

opt = optim.SGD(param,lr=0.1)


nsample = 2
nparticle = 3
dim = 1

q0 = torch.rand([nsample*nparticle,dim])
q1 = torch.rand([nsample*nparticle,dim])
p0 = torch.rand([nsample*nparticle,dim])
p1 = torch.rand([nsample*nparticle,dim])
p2 = torch.rand([nsample*nparticle,dim])

q_list = [q0] #,q1]
p_list = [p0] #,p1,p2]

qp_list = q_list + p_list

qp = torch.cat(qp_list,dim=-1)

print('qp shape ',qp.shape)
'''




