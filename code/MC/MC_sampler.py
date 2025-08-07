import sys
sys.path.append( '../')

from parameters.MC_parameters import MC_parameters
from MC.metropolis_mc import metropolis_mc
from MD.momentum_sampler import momentum_sampler
from data_loader.data_io import data_io
from utils.system_logs import system_logs
from utils.mydevice import mydevice

import os
import sys
import torch

import numpy as np
import random

if __name__=='__main__':
    # run something like this
    # python MC_sampler.py MC_config.dict train

    argv = sys.argv
    MCjson_file = argv[1]
    mode = argv[2]

    MC_parameters.load_dict(MCjson_file)

    seed        = MC_parameters.seed        # set different seed for generate data (train, valid, and test)
    interval    = MC_parameters.interval    # take mc steps every given interval
    nparticle   = MC_parameters.nparticle
    boxsize     = MC_parameters.boxsize

    # io varaiables
    filename    = MC_parameters.filename

    # seed setting
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    torch.set_default_dtype(torch.float64)

    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    data_io_obj = data_io()
    metropolis_mc   = metropolis_mc(system_logs)

    if mode == 'train':

        q_hist, U, ACCRatio, spec = metropolis_mc.step()
        # take q every interval
        q_hist = q_hist[:, 0::interval, :, :]


    elif mode == 'test':

        q_hist, U, ACCRatio, spec = metropolis_mc.step()
        q_hist = q_hist[:, -1, :, :]
        q_hist = torch.unsqueeze(q_hist,dim=1)

    else:
        assert (False), 'invalid mode ....'

    # shape is  [mc nsamples, mc interval step, nparticle, DIM]
    q_hist = torch.reshape(q_hist, (-1, q_hist.shape[2], q_hist.shape[3]))
    # shape is  [(mc nsamples x mc step), nparticle, DIM]

    # momentum sampler
    momentum_sampler = momentum_sampler(q_hist.shape[0])
    p_hist = momentum_sampler.momentum_samples()
    # shape is  [nsamples, nparticle, DIM]   ; nsamples = mc nsamples x mc step

    bs_tensor = torch.zeros(p_hist.shape)
    bs_tensor.fill_(boxsize)
    # bs_tensor.shape is [nsamples, nparticle, DIM]

    qpl_list = torch.stack((q_hist,p_hist,bs_tensor),dim=1)
    # shape is [nsamples, (q, p, boxsize), nparticle, DIM]

    qpl_list = torch.unsqueeze(qpl_list,dim=2)
    # shape is [nsamples, 3=(q, p, boxsize), 1, nparticle, DIM]

    data_io_obj.write_trajectory_qpl(filename, qpl_list)
    print('file write dir:', filename, 'shape', qpl_list.shape)

    ########### analysis ###########
    json_dir_name = os.path.dirname(filename)  + '/'

    if not os.path.exists(json_dir_name):
                os.makedirs(json_dir_name)

    print('file write analysis dir:', json_dir_name)
    torch.save((U, ACCRatio, spec), json_dir_name + f'n{nparticle}rho{MC_parameters.rho}T{MC_parameters.temperature}_U_ACCRatio_spec.pt')
    ################################

    system_logs.print_end_logs()
