from utils.pbc import delta_state
from utils.pbc import delta_pbc
from utils.mydevice import mydevice
from ML.force_functions.fbase import fbase
import torch

class pw_base(fbase):

    def __init__(self,net1,net2):
        super().__init__(net1,net2)
        print('pw fnn')

    def evalall(self,net,q_list,p_list,l_list,tau):

        # prepare input to feed into network
        x = self.prepare_input(q_list,p_list,l_list,tau)
        y = net(x)
        dim = net.output_dim
        y_shape = torch.Size((q_list.shape[0],q_list.shape[1],dim))
        y2 = self.unpack_dqdp_tau(y, y_shape)
        return y2

    def eval1(self,q_list,p_list,l_list,tau):
        return self.evalall(self.net1,q_list,p_list,l_list,tau)

    def eval2(self,q_list,p_list,l_list,tau):
        return self.evalall(self.net2,q_list,p_list,l_list,tau)


    def unpack_dqdp_tau(self,y, qlist_shape):
        nsamples, nparticle, DIM = qlist_shape
        y1 = torch.reshape(y, (nsamples, nparticle, nparticle, DIM))
        dy = torch.diagonal(y1,0,1,2) # offset, nparticle, nparticle
        torch.fill_(dy,0.0)
        y2 = torch.sum(y1, dim=2)
        # y2.shape = [nsamples,nparticle,2]
        return y2

    # prepare pairwise
    def prepare_input(self,q_list,p_list,l_list,tau):
        nsamples, nparticle, DIM = q_list.shape

        dq = delta_pbc(q_list, l_list)
        # shape is [nsamples, nparticle, nparticle, DIM]
        dq = torch.reshape(dq, (nsamples * nparticle * nparticle, DIM))
        # shape is [nsamples* nparticle* nparticle, DIM]

        dp = delta_state(p_list)

        # dq.shape = dp.shape = [nsamples, nparticle, nparticle, 2]
        dp = torch.reshape(dp, (nsamples * nparticle * nparticle, DIM))
        # shape is [nsamples* nparticle* nparticle, DIM]

        tau_tensor = torch.zeros([nsamples*nparticle*nparticle, 1],requires_grad=False) + 0.5*tau
        tau_tensor = mydevice.load(tau_tensor)

        #tau_tensor.fill_(tau * 0.5)  # tau_tensor take them tau/2

        x = torch.cat((dq, dp, tau_tensor), dim=-1)
        # dqdp.shape is [ nsamples*nparticle*nparticle, 5]

        return x
