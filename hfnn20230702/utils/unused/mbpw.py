import torch

from utils.mydevice import mydevice
import itertools

class mbpw:

    def __init__(self,mb,pw):
        self.mb = mb
        self.pw = pw

    def parameters(self):
        return itertools.chain(self.mb.parameters(),self.pw.parameters())

    def tau_parameters(self):
        return [self.mb.tau,self.pw.tau]

    def get_tau(self,tau_id):
        mbtau = self.mb.tau[tau_id]
        pwtau = self.pw.tau[tau_id]
        return 0.5*(torch.abs(mbtau)+torch.abs(pwtau))

    def unpack_mbpw_input(self,input_list):
        inputmb,inputpw = zip(*input_list)
        inputmb = list(inputmb)
        inputpw = list(inputpw)
        return inputmb,inputpw

    def eval(self,q_input_list,p_input_list):
        #print('mbpw.py eval here')
        q0mb, q0pw = self.unpack_mbpw_input(q_input_list)
        p0mb, p0pw = self.unpack_mbpw_input(p_input_list)
        return  self.mb.eval(q0mb,p0mb) + self.pw.eval(q0pw,p0pw)

    def prepare_q_input(self,pwnet_id,q_list,p_list,l_list):
        qmb = self.mb.prepare_q_input(pwnet_id,q_list,p_list,l_list)
        qpw = self.pw.prepare_q_input(pwnet_id,q_list,p_list,l_list)
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

    def max_abs_grad(self,name): # SJ update
        mbmax_grads = self.mb.max_abs_grad(name)
        pwmax_grads = self.pw.max_abs_grad(name)
        return max(mbmax_grads, pwmax_grads)

