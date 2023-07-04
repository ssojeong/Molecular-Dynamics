import sys
sys.path.append('../../')

import numpy as np
import itertools
import torch
import math
from utils.system_logs           import system_logs
from utils.mydevice              import mydevice
from utils.pbc                      import single_particle_dq_pbc
from hamiltonian.lennard_jones2d    import lennard_jones2d
from MD.velocity_verlet_MD          import velocity_verlet_MD
import matplotlib.pyplot as plt


def pack_data(qpl_list):

    q_traj = qpl_list[:,0,0,:,:].clone().detach()
    p_traj = qpl_list[:,1,0,:,:].clone().detach()
    l_init = qpl_list[:,2,0,:,:].clone().detach()

    l_init = mydevice.load(l_init)
    q_init = mydevice.load(q_traj)
    p_init = mydevice.load(p_traj)

    return q_init,p_init,l_init

def dq_rand(q_st,q_rand,l_list):

    nsamples, traj, nparticles, DIM = q_st.shape
    # shape = [nsamples, trajectory, nparticles, DIM]

    dq_rand = single_particle_dq_pbc(q_st, q_rand, l_list)
    dq_rand_sq = torch.sqrt(torch.sum(dq_rand * dq_rand, dim=3))
    mean_dq_rand_sq_par = torch.sum(dq_rand_sq, dim=2) / nparticles
    # scaling particle boxsize
    mean_dq_rand_sq_par = mean_dq_rand_sq_par / torch.mean(l_list)

    mean_dq_rand_sq = torch.sum(mean_dq_rand_sq_par, dim=0) / nsamples
    # shape = [trajectory]
    
    return mean_dq_rand_sq

def dqdp(traj_st, traj_lt):

    nsamples, _, traj, nparticles, DIM = traj_st.shape
    # shape = [nsamples, (q,p,boxsize), trajectory, nparticles, DIM]

    q_st = traj_st[:, 0, :, :, :] ; q_lt = traj_lt[:, 0, :, :, :]
    p_st = traj_st[:, 1, :, :, :] ; p_lt = traj_lt[:, 1, :, :, :]
    l_list = traj_st[:, 2, :, :, :]
    # shape = [nsamples, trajectory, nparticles, DIM]
    #print('q st shape', q_st.shape)

    dq = single_particle_dq_pbc(q_st, q_lt, l_list)
    dp = p_st - p_lt
    #print('dq shape', dq.shape)
    
    dq_sq = torch.sqrt(torch.sum(dq * dq, dim=3))
    dp_sq = torch.sqrt(torch.sum(dp * dp, dim=3))
    # shape = [nsamples, trajectory, nparticles]

    mean_dq_sq_par = torch.sum(dq_sq, dim=2) / nparticles
    mean_dp_sq_par = torch.sum(dp_sq, dim=2) / nparticles
    # shape = [nsamples,trajectory]

    # scaling particle boxsize
    mean_dq_sq_par = mean_dq_sq_par / torch.mean(l_list)
    mean_dp_sq_par = mean_dp_sq_par/ torch.mean(l_list)

    mean_dq_sq = torch.sum(mean_dq_sq_par, dim=0) / nsamples
    mean_dq_sq2 = torch.sum(mean_dq_sq_par * mean_dq_sq_par , dim=0)/ nsamples
    std_dq_sq = (mean_dq_sq2 - mean_dq_sq * mean_dq_sq)**0.5 / math.sqrt(nsamples)

    mean_dp_sq = torch.sum(mean_dp_sq_par, dim=0) / nsamples
    mean_dp_sq2 = torch.sum(mean_dp_sq_par * mean_dp_sq_par , dim=0)/ nsamples
    std_dp_sq = (mean_dp_sq2 - mean_dp_sq * mean_dp_sq)**0.5 / math.sqrt(nsamples)

    mean_dqdp_sq = torch.sum(mean_dq_sq_par, dim=0) / nsamples + torch.sum(mean_dp_sq_par, dim=0) / nsamples
    mean_dqdp_sq2 = torch.sum((mean_dq_sq_par + mean_dp_sq_par)*(mean_dq_sq_par + mean_dp_sq_par), dim=0)/ nsamples
    std_dqdp_sq = (mean_dqdp_sq2 - mean_dqdp_sq * mean_dqdp_sq)**0.5 / math.sqrt(nsamples)
    # shape = [trajectory]

    return mean_dq_sq, std_dq_sq, mean_dp_sq, std_dp_sq, mean_dqdp_sq, std_dqdp_sq

def l_max_distance(l_list):
    boxsize = torch.mean(l_list)
    L_h = boxsize / 2.
    q_max = math.sqrt(L_h * L_h + L_h * L_h)
    print('boxsize', boxsize.item(), 'maximum distance dq = {:.2f}, dq^2 = {:.2f}'.format(q_max, q_max * q_max))
    return boxsize, q_max

if __name__ == '__main__':
    # root square distance of positions and momentum

    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs() 

    argv = sys.argv
    if len(argv) != 7:
        print('usage <programe> <npar> <rho> <temp> <level> <tau max> <name>' )
        quit()

    param1 = argv[1]
    param2 = argv[2]
    param3 = argv[3]
    param4 = argv[4]
    param5 = float(argv[5])
    param6 = argv[6]

    states = { "npar" : param1,
               "rho"  : param2,
               "T"    : param3,
               "level" : param4,
               "tau_max": param5,
               "name" : param6
              }

    npar = states["npar"]
    rho = states["rho"]
    T = states["T"]
    level = states["level"]
    t_max = states["tau_max"]

    data = { "st_file" : '../../../data_sets/gen_by_MD/noML-metric-st1e-4every0.1t100/n{}rho{}T{}/'.format(npar,rho,T)
             + 'n{}rho{}T{}.pt'.format(npar,rho,T),
             #'lt_file' : '../../../data_sets/gen_by_MD/noML-metric-lt0.001every0.1t0.7t100/n{}rho{}T{}/'.format(npar,rho,T)
             #+ 'n{}rho{}T{}.pt'.format(npar,rho,T)
             'lt_file': '../../../data_sets/gen_by_ML/lt0.1dpt1800000/n{}rho{}T{}/'.format(npar, rho, T) + "{}".format(states["name"])
             }

    maindict = {
            "traj_len": 8,
            "everytau": 0.1,  # md,ml = 0.1 at tau=0.1 or 0.2 at tau=0.2 
            "tau_long": 0.1,
            "everysave": 0.1
		}

    print('load file', data, flush=True)
    data1 = torch.load(data["st_file"])
    data2 = torch.load(data["lt_file"])

    qpl_traj1 = data1['qpl_trajectory']
    qpl_traj2 = data2['qpl_trajectory']
    # shape [nsamples, 3, traj, npar, dim]
    print('load qpl traj ', qpl_traj1.shape, qpl_traj2.shape)
 
    _, _, _, nparticle, _ = qpl_traj1.shape

    traj_len = maindict["traj_len"]
    everytau = maindict["everytau"]
    everysave = maindict["everysave"]
    pair_step_idx = round(everytau/everysave)
    input_seq_idx = traj_len - 1
    nstep = round((t_max - (input_seq_idx * maindict["tau_long"])) / everysave)

    lj_obj = lennard_jones2d()
    mdvv = velocity_verlet_MD(lj_obj)

    qpl_traj_st = mydevice.load(qpl_traj1[:,:,input_seq_idx:input_seq_idx+nstep+1:pair_step_idx,:,:])
    # shape = [nsamples, (q,p,boxsize), trajectory, nparticles, DIM]
    print('qpl traj ref shape', qpl_traj_st.shape, 'level ', level, 'nsteps ', nstep, flush=True)

    qpl_traj_lt = mydevice.load(qpl_traj2[:,:,0:int(nstep/pair_step_idx)+1,:,:])
    # shape = [nsamples, (q,p,boxsize), trajectory, nparticles, DIM]
    print('qpl traj shape', qpl_traj_lt.shape,flush=True)

    q_list1, p_list1, l_list1 = pack_data(qpl_traj_st)
    # q_list2, p_list2, l_list2 = pack_data(qpl_traj_lt)
    print(q_list1.shape,p_list1.shape, l_list1.shape, flush=True)
    # print(q_list2.shape,p_list2.shape, l_list2.shape, flush=True)

    boxsize, q_max = l_max_distance(l_list1)

    q_rand = mydevice.load((torch.rand(qpl_traj_st[:,0,:,:,:].shape) - 0.5)) * boxsize
    # shape = [nsamples, trajectory, nparticles, DIM]
    print('q rand shape', q_rand.shape, flush=True)

    dq_rand = dq_rand(qpl_traj_st[:,0,:,:,:], q_rand, qpl_traj_st[:,2,:,:,:])
    assert torch.equal(qpl_traj_st[:,:,0,:,:],qpl_traj_lt[:,:,0,:,:]) == True, 'init states not match....'

    t = np.arange(0,qpl_traj_st.shape[2])
    print('t len', len(t), 'traj len', qpl_traj_lt.shape[2])
 
    q_mean12, q_std12, p_mean12, p_std12, qp_mean12, qp_std12 = dqdp(qpl_traj_st, qpl_traj_lt)

    print('len', len(q_mean12), 'tau_max ', t_max, "\n" 'avg dq', q_mean12.shape, "\n" 'std dq', q_std12.shape,
    "\n" 'avg dp', p_mean12.shape, "\n" 'std dp', p_std12.shape, 'dq rand', dq_rand.shape)

    print('dqdp', '#column:time random_q q_mean q_std p_mean p_std')
    for j in range(0,q_mean12.shape[0]):
      print('dqdp', input_seq_idx*everysave + t[j]*everytau, dq_rand[j].item(), q_mean12[j].detach().cpu().numpy(),
            q_std12[j].detach().cpu().numpy(),p_mean12[j].detach().cpu().numpy(), p_std12[j].detach().cpu().numpy())

