import sys 
sys.path.append('../../')
from utils.system_logs              import system_logs
from utils.mydevice                 import mydevice
#from utils.pbc                      import pbc
from utils.pbc                      import single_particle_dq_pbc
from hamiltonian.lennard_jones2d    import lennard_jones2d
from MD.velocity_verlet_MD          import velocity_verlet_MD
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch
import itertools
import numpy as np
import time
import math


def pack_data(qpl_list):

    q_traj = qpl_list[:,0,0,:,:].clone().detach()
    p_traj = qpl_list[:,1,0,:,:].clone().detach()
    l_init = qpl_list[:,2,0,:,:].clone().detach()

    q_traj = mydevice.load(q_traj)
    p_traj = mydevice.load(p_traj)
    l_init = mydevice.load(l_init) 

    return q_traj,p_traj,l_init

def norm_dq(q_list, q_eps_list, l_list):

    nsamples, nparticle, _ = q_list.shape
    # shape = [nsamples, nparticles, DIM]

    dq = single_particle_dq_pbc(q_list, q_eps_list, l_list)
    # shape = [nsamples, nparticles, DIM]

    dq_sqrt = torch.sqrt(torch.sum(dq * dq, dim=2))
    # shape = [nsamples, nparticles]

    mean_dq_sqrt = torch.sum(dq_sqrt, dim=1) / nparticle
    # shape = [nsamples]

    return mean_dq_sqrt

def lyapunov_exp(q_list, q_eps_list, l_list):
    dq_list = norm_dq(q_list, q_eps_list, l_list)
    # shape = [nsamples]
    return dq_list

if __name__ == '__main__':

    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    argv = sys.argv
    if len(argv) != 6:
        print('usage <programe> <npar> <rho> <temp> <tau> <thrsh>' )
        quit()

    param1 = argv[1]
    param2 = argv[2]
    param3 = argv[3]
    param4 = argv[4]
    param5 = argv[5]

    states = { "npar" : param1,
               "rho"  : param2,
               "T"    : param3,
               "tau_cur" : param4,
               "t_thrsh" : param5
              }

    npar = states["npar"]
    rho = states["rho"]
    T = states["T"]
    tau_cur = float(states["tau_cur"])
    t_thrsh = int(states["t_thrsh"])

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(34952)

    data = {"filename": "../../../data_sets/gen_by_MD/noML-metric-st1e-4every0.1t100/n{}rho{}T{}/".format(npar,rho,T)
                                       + 'n{}rho{}T{}.pt'.format(npar,rho,T),
	        "dt_init_config_filename": "../../../data_sets/gen_by_MD/noML-metric-lt0.01every0.1t0.7t100/n{}rho{}T{}/".format(npar,rho,T)
                                        + 'n{}rho{}T{}_'.format(npar, rho, T),
            "saved_filename" : "../../../data_sets/gen_by_MD/noML-metric-lt0.01every0.1t0.7t100/n{}rho{}T{}/".format(npar,rho,T)
                             + 'n{}rho{}T{}tau{}_wolf.pt'.format(npar, rho, T,tau_cur)
            }

    maindict = {"dt"  : 1e-4,
                "eps" : 2e-4,
                "traj_len": 8,
	            "dt_samples" : 1, # no. of eps samples
                "therm_state": 'n{}rho{}T{}'.format(npar, rho, T),
	            "t_thrsh" : t_thrsh, # 1000 # if short time step, t_thrsh give more
                "t_max" : 51
                } #101

    t_max = maindict["t_max"]
    nsteps = round(t_max / tau_cur)   # no. of timesteps in trajectories
    t_thrsh = maindict["t_thrsh"] # thrsh not over t incremented until eps
    saved_filename = data["saved_filename"]
    dt_init_config_filename = data["dt_init_config_filename"]
    therm_state = maindict["therm_state"]
    traj_len = maindict["traj_len"]
    input_seq_idx = traj_len - 1
    dt = maindict['dt']
    eps = maindict['eps']

    print('nsteps ', nsteps, 't thrsh ', t_thrsh)
    assert (nsteps >= t_thrsh), '....incompatible nsteps and t thrsh'

    lj_obj = lennard_jones2d()
    mdvv = velocity_verlet_MD(lj_obj)

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    for j in range(0, maindict['dt_samples']):
        print('sample for dt traj ====' , j )
        print('t max ', maindict["t_max"] )
        print('nsteps ', nsteps)

        thrsh = [0]
        R_sample_append = []
        dq_append = []
        avg_dq_append = []

        for i in range(nsteps):

            print('t thrsh ======', thrsh, flush=True)
            taccum = itertools.accumulate(thrsh)
            t_accum = list(taccum)
            print('t accum ======', t_accum)

            if t_accum[-1] >= nsteps:
                    print('accurated t reach t maximum... finish running ....')
                    break

            if i == 0:

                print('start t =======', i)
                qpl_list = torch.load(data["filename"], map_location=map_location)
                qpl_init  = qpl_list['qpl_trajectory'][:,:,input_seq_idx:input_seq_idx+1,:,:]
                #shape [nsamples, 3, 1, nparticles, dim]
                print('load data init file : ',  qpl_init.shape)
                quit()
                qpl_dt_init = qpl_init + mydevice.load(torch.FloatTensor(qpl_init.shape).uniform_(-maindict['dt'], maindict['dt']))
                # shape [nsamples,3,1,nparticles,dim]

                q_init, p_init, l_list = pack_data(qpl_init)
                q_dt_init, p_dt_init, _ = pack_data(qpl_dt_init)

                data1 = { 'qpl_trajectory':qpl_dt_init, 'tau_short':tau_cur, 'tau_long': tau_cur }
                print('qpl traj ', qpl_dt_init.shape, 'tau short', tau_cur, 'tau_long' , tau_cur, flush=True)
                torch.save(data1, dt_init_config_filename + 'eps_{}.pt'.format(i))
                print('save dt init config filename :', dt_init_config_filename + 'dt_{}.pt'.format(i))

                dq_init = lyapunov_exp(q_init, q_dt_init, l_list)
                # shape = [nsamples] 1 is initial

                avg_dq_init =  torch.sum(dq_init, dim=0) /  dq_init.shape[0]
                print('sample avg init dq list', avg_dq_init)

                l1_sample = dq_init  # shape [nsamples]
                l1_sample_avg = avg_dq_init  # shape []

                dq_append.append(dq_init) # shape [nsamples]
                avg_dq_append.append(l1_sample_avg.item()) # shape []

                q_cur = q_init
                p_cur = p_init
                q_dt_cur = q_dt_init
                p_dt_cur = p_dt_init

            else:

                print('start t =======', t_accum[-1])
                qpl_init  = torch.stack((q_cur,p_cur,l_list),dim=1)  #shape [nsamples, 3, nparticles, dim]
                qpl_init = torch.unsqueeze(qpl_init, dim=2)
                # shape [nsamples, 3, 1, nparticles, dim]
                #print('t iter', i, q_cur[0])

                qpl_dt_init = qpl_init + mydevice.load(torch.FloatTensor(qpl_init.shape).uniform_(-maindict['dt'], maindict['dt']))

                q_init, p_init, l_list = pack_data(qpl_init)
                q_dt_init, p_dt_init, _ = pack_data(qpl_dt_init)

                dq_init = lyapunov_exp(q_init, q_dt_init, l_list) # along time

                avg_dq_init =  torch.sum(dq_init, dim=0) /  dq_init.shape[0]
                print('sample avg init dq list', avg_dq_init)

                l1_sample = dq_init  # shape [nsamples]
                l1_sample_avg = avg_dq_init # shape []

                q_cur = q_init
                p_cur = p_init
                q_dt_cur = q_dt_init
                p_dt_cur = p_dt_init

            for k in range(t_thrsh): # thrsh not over t incremented until eps

                    print('increment t until eps =======', k+1, flush=True)
                    #print('before iter', q_cur[0])
                    q_list1, p_list1, l_list1 = mdvv.one_step(q_cur, p_cur, l_list, tau_cur)
                    q_list2, p_list2, l_list2 = mdvv.one_step(q_dt_cur, p_dt_cur, l_list, tau_cur)
                    #print(qpl_list1.shape, qpl_list2.shape)

                    dq_list = lyapunov_exp(q_list1, q_list2, l_list1) # along time
                    # shape [nsmaples]

                    #print('GPU memory % allocated:', round(torch.cuda.memory_allocated(0)/1024**3,2) ,'GB', '\n')
                    avg_dq_list =  torch.sum(dq_list, dim=0) /  dq_list.shape[0]
                    # shape []

                    q_cur = q_list1
                    p_cur = p_list1
                    l_list = l_list1
                    q_dt_cur = q_list2
                    p_dt_cur = p_list2
                    #print('after iter',q_cur[0])

                    if avg_dq_list < eps: # 4e-2 4e-3
                        print('L < eps .....') #, avg_dq_list)

                        if k+1 == t_thrsh:

                            print('Reach the end of t thrsh .....') #, avg_dq_list)
                            l2_sample = dq_list  # shape [nsamples]]
                            l2_sample_avg = avg_dq_list  # shape []

                            R_sample = l2_sample / l1_sample # shape [nsamples]
                            R_avg = l2_sample_avg / l1_sample_avg # shape []

                            thrsh.append(k+1)
                            R_sample_append.append(R_sample)
                            avg_dq_append.append(l2_sample_avg.item())

                        else:
                            continue

                    else:

                        print('L > eps .....', avg_dq_list.item(), 't =', k+1)
                        l2_sample = dq_list  # shape [nsamples]
                        l2_sample_avg = avg_dq_list  # shape []

                        R_sample = l2_sample / l1_sample # shape [nsamples]
                        R_avg = l2_sample_avg / l1_sample_avg # shape []

                        thrsh.append(k+1)
                        R_sample_append.append(R_sample)
                        avg_dq_append.append(l2_sample_avg.item())
                        #print('increment t ===== ', k+1, l1, LogR)
                        break

            print('num of loop', i+1)

        #LogR_sample_sum = torch.sum(torch.stack(LogR_sample_append,dim=1),dim=1) # shape [nsamples]
        #LogR_avg_sum = torch.sum(torch.Tensor(LogR_avg_append)) # shape []
        print(t_accum, R_sample_append)

        data = {'t_accum' : t_accum, 'R_sample_append' : R_sample_append}
        torch.save(data, saved_filename)
        print('save file dir', saved_filename)
  
system_logs.print_end_logs()

