import torch

from utils.mydevice import mydevice

class mbpw:

    def __init__(self,mb,pw):
        self.mb = mb
        self.pw = pw
        self.alpha1 = torch.tensor([-1.35],requires_grad=True,device=mydevice.get()) # start with 12% mb
        self.alpha2 = torch.tensor([-1.35],requires_grad=True,device=mydevice.get()) # start with 12% mb

    def parameters(self):
        return [self.alpha1,self.alpha2]

    def eval1(self,q_list,p_list,l_list,tau):
        a1 = torch.sigmoid(self.alpha1)
        return  a1     *self.mb.eval1(q_list,p_list,l_list,tau)+\
                (1.-a1)*self.pw.eval1(q_list,p_list,l_list,tau)

    def eval2(self,q_list,p_list,l_list,tau):
        a2 = torch.sigmoid(self.alpha2)
        return  a2     *self.mb.eval2(q_list,p_list,l_list,tau)+\
                (1.-a2)*self.pw.eval2(q_list,p_list,l_list,tau)

    def a(self):
        return torch.sigmoid(self.alpha1).item(), torch.sigmoid(self.alpha2).item()


    def clamp_a(self,alp):
        lower_thrsh = torch.tensor([-5.0],requires_grad=True,device=mydevice.get())  # 0.9991 use pw net
        upper_thrsh = torch.tensor([ 2.2],requires_grad=True,device=mydevice.get())  
          # 0.9002 use mb net -- found that mb net can't be optimized well
        if alp.data<lower_thrsh.data: alp.data = lower_thrsh.data
        if alp.data>upper_thrsh.data: alp.data = upper_thrsh.data
        
    def clamp_alpha(self):
        self.clamp_a(self.alpha1)
        self.clamp_a(self.alpha2)

    def state_dict(self):
        return { 'alpha1': self.alpha1, 'alpha2': self.alpha2 }

    def load_state_dict(self,state):
        self.alpha1.data = state['alpha1'].data
        self.alpha2.data = state['alpha2'].data





