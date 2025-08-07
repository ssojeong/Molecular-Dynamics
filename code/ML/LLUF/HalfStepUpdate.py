import torch
import numpy as np
import torch.nn as nn
from utils.mydevice import mydevice

from ML.LLUF.SingleParticle import SingleParticle
from ML.LLUF.MultiParticles import MultiParticles
from ML.LLUF.ReadoutStep import ReadoutStep
from utils.force_stat          import force_stat


class HalfStepUpdate(nn.Module):

    def __init__(self, prepare_data_obj, single_particle_net, multi_particle_net, readout_step_net, t_init, nnet=1):
        super().__init__()

        self.prepare_data = prepare_data_obj  # prepare_data object
        self.single_par = SingleParticle(single_particle_net)
        self.multi_par = MultiParticles(multi_particle_net)
        self.update_step = ReadoutStep(readout_step_net)

        self.tau_init = np.random.rand(nnet) * t_init  # change form 0.01 to 0.001
        self.tau = nn.Parameter(torch.tensor(self.tau_init, device=mydevice.get()))
        self.f_stat = force_stat()

    # for update q <- q + tau[2]*p + f_q
    # see LLUF_MD for use of this function

    def forward(self,q_input_list,p_input_list,q_prev):
        x = self.prepare_data.cat_qp(q_input_list,p_input_list)
        # shape [nsamples, nparticles, traj_len, ngrids * DIM * (q,p)]
        x = self.single_par.eval(x)
        # shape [nsample, nparticle, embed_dim]
        x = self.multi_par.eval(x, q_prev)
        # shape=[nsample,nparticle,embed_dim]
        x = self.update_step.eval(x)
        # shape=[nsample,nparticle,dim=2]
        self.f_stat.accumulate(x)
        return x * torch.abs(self.tau) # return the update step


