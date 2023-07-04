from hamiltonian.noML_hamiltonian import noML_hamiltonian
from phase_space import phase_space
from utils.data_io import data_io
from parameters.MC_parameters import MC_parameters
from integrator.metropolis_mc import metropolis_mc
from integrator.momentum_sampler import momentum_sampler
from utils.system_logs import system_logs
from utils.device import mydevice

import os
import sys
import torch

import numpy as np
import random

if __name__=='__main__':
    # run something like this
    # python MC_sampler.py ../data/gen_by_MC/basename/MC_config.dict

    argv = sys.argv
    MCjson_file = argv[1]
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

    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    data_io_obj     = data_io()
    phase_space     = phase_space.phase_space()
    noMLhamiltonian = noML_hamiltonian()

    metropolis_mc   = metropolis_mc(system_logs)

    q_hist, U, ACCRatio, spec = metropolis_mc.step(noMLhamiltonian, phase_space)

    # take q every interval
    q_hist = q_hist[:, 0::interval, :, :]
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
    print('file write dir:', filename)

    ########### analysis ###########
    # json_dir_name = '../data/gen_by_MC/n{}rho{}/'.format(nparticle, MC_parameters.rho) + 'anaysis/'

    # if not os.path.exists(json_dir_name):
    #             os.makedirs(json_dir_name)

    # torch.save((ACCRatio, spec), '../data/gen_by_MC/n{}rho{}/analysis/n{}T{}_ACCRatio_spec.pt'.format(nparticle, MC_parameters.rho, nparticle, MC_parameters.temperature))
    ################################

    system_logs.print_end_logs()