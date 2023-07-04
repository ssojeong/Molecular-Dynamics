from checkpoint import checkpoint

import torch
import itertools 
import torch.nn    as nn
import torch.optim as optim

class net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1,1)

    def forward(self,x):
        return self.fc(x)

    def print_param(self):
        print(self.fc.weight,self.fc.bias)



def print_param(net_list):

    for idx,n in enumerate(net_list):
        print('------ net ------ ',idx)
        n.print_param()


if __name__=='__main__':

    net1 = net()
    net2 = net()
    net3 = net()

    net_list = [net1,net2,net3]
    param = itertools.chain(net1.parameters(),net2.parameters(),net3.parameters())
    opt = optim.SGD(param,lr=1e-2)
    sch = optim.lr_scheduler.StepLR(opt,10,0.1)

    ckpt = checkpoint(net_list,opt,opt,sch)

    batch = 2
    xb = torch.rand([batch,1])

    print_param(net_list)


    for e in range(100):

        opt.zero_grad()
        x = xb.clone().detach()
        for n in net_list:
            x = n(x)

        loss = torch.sum(x*x)
        loss.backward()
        opt.step()

    print('loss ',loss)
    print_param(net_list)

    ckpt.save_checkpoint('t.log_everyepoch')

    ckpt.load_checkpoint('t.log_everyepoch')
 
    print_param(net_list)






