import sys
sys.path.append( '../../')

import itertools
import torch
import numpy as np
from utils.system_logs              import system_logs
from utils.mydevice                 import mydevice
from hamiltonian.lennard_jones2d    import lennard_jones2d

def pack_data(qpl_list, idx):

    q_init = qpl_list[:,0,idx,:,:].clone().detach()
    p_init = qpl_list[:,1,idx,:,:].clone().detach()
    l_init = qpl_list[:,2,idx,:,:].clone().detach()

    q_init = mydevice.load(q_init)
    p_init = mydevice.load(p_init)
    l_init = mydevice.load(l_init)

    return q_init,p_init,l_init

def total_energy(potential_function, q_list, p_list, l_list):
    pe = potential_function.total_energy(q_list, l_list)
    ke = torch.sum(p_list * p_list, dim=(1, 2)) * 0.5
    return ke, pe


if __name__ == '__main__':
    # python e_conserve.py 16 0.025 0.48 40 g

    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(34952)

    argv = sys.argv
    if len(argv) != 9:
        print('usage <programe> <npar> <rho> <temp> <nsteps> <gamma> <saved_model> <dpt> <region>' )
        quit()

    npar = argv[1]
    rho = argv[2]
    T = argv[3]
    nstep = int(argv[4])
    gamma = int(argv[5])
    saved_model = argv[6]
    dpt = int(argv[7])
    region = argv[8]

    data = { 
             #"filename": '../../../data_sets/gen_by_MD/noML-metric-st1e-4every0.1t100/n{}rho{}T{}/'.format(npar,rho,T) + 'n{}rho{}T{}.pt'.format(npar,rho,T)}
             #"filename" : '../../../data_sets/gen_by_MD/noML-metric-lt0.01every0.1t0.7t100/n{}rho{}T{}/'.format(npar,rho,T) +
             #               'n{}rho{}T{}gamma{}.pt'.format(npar,rho,T,gamma)}
             "filename" : '../../../data_sets/gen_by_ML/lt0.1dpt{}_{}/n{}rho{}T{}/'.format(dpt,region,npar,rho,T) + 'pred_n{}len08ws08gamma{}mb{}_tau0.1.pt'.format(npar,gamma,saved_model) }


    maindict = { #"save_dir" : "../../../data_sets/gen_by_MD/noML-metric-st1e-4every0.1t100/n{}rho{}T{}/energytmax100.pt".format(npar,rho,T)
                 #"save_dir" : "../../../data_sets/gen_by_MD/noML-metric-lt0.01every0.1t0.7t100/n{}rho{}T{}/energy_gamma{}_tmax100.pt".format(npar,rho,T,gamma)}
                 "save_dir" : "../../../data_sets/gen_by_ML/lt0.1dpt{}_{}/n{}rho{}T{}/".format(dpt,region,npar,rho,T) + 'energy_gamma{}mb{}_nsteps{}.pt'.format(gamma,saved_model,nstep)}

    print(data)
    lj = lennard_jones2d()

    data1 = torch.load(data["filename"],map_location=map_location)
    qpl_traj = data1['qpl_trajectory']
    print('traj shape ', qpl_traj.shape)

    print('load filename ..', data["filename"])
    print('calc energy .......')

    tot_u_append = []
    tot_k_append = []
    tot_e_append = []
    for t in range(qpl_traj.shape[2]): # trajectory length
      q_list, p_list, l_list = pack_data(qpl_traj, t)
      tot_k, tot_u = total_energy(lj, q_list, p_list, l_list) # shape [nsamples]
      # print('t===', t, 'q', q_list.min().item(), q_list.max().item(), 'p', p_list.min().item(), p_list.max().item(), 'u', min(tot_u).item(),max(tot_u).item(), 'k', min(tot_k).item(), max(tot_k).item())
      tot_u_append.append(tot_u)
      tot_k_append.append(tot_k)
      tot_e_append.append(tot_u+tot_k)

    print('finished calc energy .....')
    tot_u_append = torch.stack(tot_u_append, dim=0)  # shape [trajectory, nsamples]
    tot_k_append = torch.stack(tot_k_append, dim=0)  # shape [trajectory, nsamples]
    tot_e_append = torch.stack(tot_e_append,dim=0) # shape [trajectory, nsamples]
    print(torch.min(tot_u_append).item(), torch.max(tot_u_append).item())

    torch.save({'pe':tot_u_append, 'ke':tot_k_append, 'energy': tot_e_append},maindict["save_dir"])
    print('save dir ..', maindict["save_dir"])

