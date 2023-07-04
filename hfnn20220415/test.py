from ML.force_functions.mbpw  import mbpw 
from ML.force_functions.mb_ff import mb_ff
from ML.force_functions.pw_ff import pw_ff
from MD.velocity_verlet3      import velocity_verlet3
from utils.mydevice           import mydevice
from utils.pbc                import pbc
from ML.trainer.loss          import loss

import numpy       as np
import torch.optim as optim
import torch.nn    as nn
import torch

class potential_function:
    def total_energy(self,q,l):
        return torch.sum(q*q,dim=(1,2))


class nnet(nn.Module):

    def __init__(self,input_dim,output_dim):
        super().__init__()
        h = max(4*input_dim,128)
        self.fc1 = nn.Linear(input_dim,h)
        self.fc2 = nn.Linear(h,h)
        self.fc3 = nn.Linear(h,output_dim)
        self.output_dim = output_dim

    def forward(self,x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        return x

   
def euler(q,p,l):
    dt = 1e-3
    for t in range(100):
        a  = q.clone().detach() # ideal gas
        q  = q + dt*p # move in striaght line
        pbc(q,l)
        p  = p + dt*a # acceleration
    return q,p


if __name__=='__main__':

    torch.manual_seed(12487)
    torch.autograd.set_detect_anomaly(True)
    _ = mydevice()
    nsamples = 500
    nparticles = 16
    dim = 2
    q_init_list = []
    p_init_list = []
    l_init_list = []
    q_label_list = []
    p_label_list = []

    nbatches=1000

    for n in range(nbatches):
        q_init = torch.rand([nsamples,nparticles,dim])
        p_init = torch.rand([nsamples,nparticles,dim])-0.5
        l_init = torch.ones([nsamples,nparticles,dim]) 
        q_label,p_label = euler(q_init,p_init,l_init)
        q_init_list.append(q_init)
        p_init_list.append(p_init)
        l_init_list.append(l_init)
        q_label_list.append(q_label)
        p_label_list.append(p_label)

    ngrids = 6
    b = 0.2
    force_clip = 1.0
    nchain = 1

    pwnet_list = [nnet( 2,2),nnet( 3,2),nnet( 4,2),nnet( 5,2),nnet( 6,2),nnet( 7,2)]
    mbnet_list = [nnet(24,2),nnet(36,2),nnet(48,2),nnet(60,2),nnet(72,2),nnet(84,2)]
    pw4mb_list = [nnet(2,2),nnet(2,2),nnet(2,2)]

    for pw,mb in zip(pwnet_list,mbnet_list):
        pw = mydevice.load(pw)
        mb = mydevice.load(mb)
    for pw4mb in pw4mb_list:
        pw4mb = mydevice.load(pw4mb)

    mb = mb_ff(mbnet_list,pw4mb_list,ngrids,b,force_clip,nsamples,nparticles)
    pw = pw_ff(pwnet_list,force_clip,nsamples,nparticles)

    mbpw_obj = mbpw(mb,pw)
    vv3 = velocity_verlet3(mbpw_obj)

    lr = 1e-4
    opt  = optim.SGD(mbpw_obj.parameters(),lr)
    opt2 = optim.SGD(mbpw_obj.tau_parameters(),lr*1e-1)
    sch  = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.9)

    energy = potential_function()
    poly_deg = 2
    loss_obj = loss(energy,0.0,poly_deg)

    nepoch = 100000
    for e in range(nepoch):

        data_id = np.random.randint(0,nbatches)

        q_init = q_init_list[data_id]
        p_init = p_init_list[data_id]
        l_init = l_init_list[data_id]
        q_label = q_label_list[data_id]
        p_label = p_label_list[data_id]

        q_init = mydevice.load(q_init)
        p_init = mydevice.load(p_init)
        l_init = mydevice.load(l_init)
        q_label = mydevice.load(q_label)
        p_label = mydevice.load(p_label)

        opt.zero_grad()
        opt2.zero_grad()

        q_pred, p_pred, _ = vv3.one_step(q_init,p_init,l_init)

        loss_value = loss_obj.eval(q_pred,p_pred,q_label,p_label,q_init,p_init,l_init)

        loss_value.backward()
        opt.step()
        opt2.step()
        sch.step()
        if e%100==0:
            lr = opt.param_groups[0]['lr']
            #print('epoch',e,'loss ',loss_value.item())
            loss_obj.verbose(e,lr,'train')
            mbpw_obj.verbose(e,'train')


   

