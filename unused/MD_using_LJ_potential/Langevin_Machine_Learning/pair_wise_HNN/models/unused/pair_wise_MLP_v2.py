import torch
import torch.nn as nn
from MD_using_LJ_potential.Langevin_Machine_Learning.pair_wise_HNN.pair_wise_HNN import pair_wise_HNN
from MD_using_LJ_potential.Langevin_Machine_Learning.Integrator.ML_linear_integrator import ML_linear_integrator
import numpy as np

class pair_wise_MLP(nn.Module):

    def __init__(self, n_input, n_hidden):

        super(pair_wise_MLP, self).__init__()
        self.correction_term = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 2)
        )

    def forward(self, data, **_setting): # data -> del_list ( del_qx, del_qy, del_px, del_py, t )

        MLdHdq_ = self.correction_term(data)
        MLdHdq_ = MLdHdq_.reshape(3,2,2)  # N_particle, N_particle-1, DIM

        MLdHdq = torch.sum(MLdHdq_, dim=1) # ex) a,b,c three particles;  sum Fa = Fab + Fac

        # === get q_pred, p_pred from MLdHdq ===
        _pair_wise_HNN = pair_wise_HNN(_setting['hamiltonian'], MLdHdq)
        _setting['HNN'] = _pair_wise_HNN

        q_pred, p_pred = ML_linear_integrator(**_setting).integrate(multicpu=False)

        q_pred = q_pred.reshape(-1, q_pred.shape[2], q_pred.shape[3])
        p_pred = p_pred.reshape(-1, p_pred.shape[2], p_pred.shape[3])

        return (q_pred,p_pred)