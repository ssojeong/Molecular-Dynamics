import torch

from utils.mydevice import mydevice
import itertools

class mbpw:

    def __init__(self,mb,pw):
        self.mb = mb
        self.pw = pw
        tau_init = mb.tau_init.copy()
        self.tau = torch.tensor(tau_init,requires_grad=True,device = mydevice.get())


    def parameters(self):
        return itertools.chain(self.mb.parameters(),self.pw.parameters())

    def tau_parameters(self):
        return [self.mb.tau,self.pw.tau]

    def train_mode(self):
        self.mb.train_mode()
        self.pw.train_mode()

    def eval_mode(self):
        self.mb.eval_mode()
        self.pw.eval_mode()

    def get_tau(self,tau_id):
        hk - make tau learnable
        # mbtau = torch.abs(self.mb.tau[tau_id]) # hk
        # pwtau = torch.abs(self.pw.tau[tau_id]) # hk
        # return 0.5*(mbtau+pwtau)
        #return torch.abs(mbtau)

    def unpack_mbpw_input(self,input_list):
        inputmb,inputpw = zip(*input_list)
        inputmb = list(inputmb)
        inputpw = list(inputpw)
        return inputmb,inputpw

    def eval(self,netid,q_input_list,p_input_list, q_pre): # SJ coord
        #print('mbpw.py eval here')
        # q_input_list [(phi0,dq0),(phi1,dq1),(phi2,dq2),...]
        # p_input_list [(pi0,dp0),(pi1,dp1),(pi2,dp2),...]
        q0mb, q0pw = self.unpack_mbpw_input(q_input_list)
        # q0mb [phi0, phi1, phi2, ...] , q0pw [dq0, dq1, dq2, ...]
        p0mb, p0pw = self.unpack_mbpw_input(p_input_list)
        # p0mb [pi0, pi1, pi2, ...] , p0pw [dp0, dp1, dp2, ...]
        return  self.mb.eval(netid,q0mb,p0mb, q_pre) + self.pw.eval(netid,q0pw,p0pw)

    def prepare_q_input(self,net_id,q_list,p_list,l_list):
        qmb = self.mb.prepare_q_input(net_id,q_list,p_list,l_list)
        qpw = self.pw.prepare_q_input(net_id,q_list,p_list,l_list)
        return qmb,qpw

    def prepare_p_input(self,q_list,p_list,l_list):
        pmb = self.mb.prepare_p_input(q_list,p_list,l_list)
        ppw = self.pw.prepare_p_input(q_list,p_list,l_list)
        return pmb,ppw

    def make_mask(self,nsamples,nparticles):
        self.mb.make_mask(nsamples,nparticles)
        self.pw.make_mask(nsamples,nparticles)

    def grad_clip(self,clip_value):
        self.mb.grad_clip(clip_value)
        self.pw.grad_clip(clip_value)

    def get_mode(self):
        return 'ff'

    def verbose(self,e,label):
        msg = label + ' mb tau'
        self.mb.verbose(e,msg)
        msg = label + ' pw tau'
        self.pw.verbose(e,msg)

    # defunct
    #def max_abs_grad(self,name): # SJ update
    #    mbmax_grads = self.mb.max_abs_grad(name)
    #    pwmax_grads = self.pw.max_abs_grad(name)
    #    return max(mbmax_grads, pwmax_grads)

