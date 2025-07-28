import sys
sys.path.append( '../')

from utils.pbc import pbc
from utils.utils import assert_nan
from hamiltonian.lennard_jones2d import lennard_jones2d
from hamiltonian.thermostat import thermostat

from utils.system_logs           import system_logs
from utils.mydevice              import mydevice

import math
import numpy as np
import shutil
import json
import time
import torch

class Langevin_MD:

    def __init__(self,lennard_jones2d):
        self.lennard_jones2d = lennard_jones2d
        print(' velocity verlet MD')
        #self.f_stat = force()
        #print(' check force .....')

        #assert(self.modef == 'ff'),'hf mode not implemented in velocity_verlet3'


    def one_step(self,q_list,p_list,l_list,tau, gamma, temp):

        p_plus = thermostat(p_list, gamma, temp, tau)
        p_list_2 = p_plus - 0.5*tau*self.lennard_jones2d.derivative(q_list,l_list)

        q_list_2 = q_list + tau*p_list_2
        q_list_new = pbc(q_list_2,l_list)

        p_minus = p_list_2 - 0.5*tau*self.lennard_jones2d.derivative(q_list_new,l_list)

        p_list_new = thermostat(p_minus, gamma, temp, tau)

        assert_nan(p_list_new)
        assert_nan(q_list_new)
        return q_list_new,p_list_new,l_list


    def nsteps(self,q_list,p_list,l_list,tau,nitr,append_strike, gamma, temp):

        assert (nitr % append_strike == 0), 'incompatible strike and nitr'

        qpl_list = []

        for t in range(nitr):
            #print('====== step ', t, flush=True)

            start_time = time.time()
            q_list,p_list,l_list = self.one_step(q_list,p_list,l_list,tau, gamma, temp)
            end_time = time.time()
            
            one_step_time = end_time - start_time
            #print('nsamples {}'.format(q_list.shape[0]), 't=',t, '{:.05f} sec'.format(one_step_time))
            #if (t+1)%50 == 0: quit()

            nxt_qpl = torch.stack((q_list, p_list, l_list), dim=1)
  
            if t%500 == 0: print('.',end='',flush=True)

            if (t + 1) % append_strike == 0:
                print('====== step ', t+1, flush=True)
                qpl_list.append(nxt_qpl)
            # print('cpu used:', psutil.cpu_percent(), '\n')
            # print('memory % used:', psutil.virtual_memory()[2], '\n')

        return qpl_list

if __name__ == '__main__':
    # python velocity_verlet_MD.py MD_config.dict 0 0.46
    argv = sys.argv
    MDjson_file = argv[1]
    gamma = float(argv[2])
    temp = float(argv[3])

    with open(MDjson_file) as f:
        data = json.load(f)

    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()  

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(34952)

    if (gamma == 0.0) or (gamma == 1.0) or (gamma == 10.0) or (gamma == 100.0) or (gamma == 1000.0):
        print('gamma {} float to int .... '.format(gamma))
        gamma = int(gamma)

    t_max =  data['tau_long'] * data['niter_tau_long']
    tau_short = np.float64(data['tau_short'])
    tau = tau_short

    tau_long = data['tau_long']
    append_strike = data['append_strike']
    save2file_strike = data['save2file_strike']
    niter_tau_short = data['niter_tau_short']

    n_out_files = niter_tau_short // save2file_strike

    print(data)
    #print('tau', tau_short, 'append_strike', append_strike, 'save2file_strike',
    #      save2file_strike, 'niter_tau_short',niter_tau_short, 'n_out_files', n_out_files)
    #assert (append_strike == round(1/tau_short)), 'incompatible strike and no. steps with tau_short'
    #assert (niter_tau_short == round(t_max/tau_short) == t_max * append_strike ), 'incompatible nitr tau short and nitr'

    MC_init_config_filename = data['MC_init_config_filename']
    MD_data_dir = data['MD_data_dir']
    MD_output_basenames = data['MD_output_basenames']

    load_data = torch.load(MC_init_config_filename)
    qpl_list = load_data['qpl_trajectory']
    pql_list = mydevice.load(qpl_list)
    
    q_init = qpl_list[:,0,0,:,:]
    p_init = qpl_list[:,1,0,:,:]
    l_init = qpl_list[:,2,0,:,:]

    potential_function = lennard_jones2d()
    mdvv = Langevin_MD(potential_function)

    q_list = q_init.clone().detach()
    p_list = p_init.clone().detach()
    l_list = l_init.clone().detach()

    #shutil.copy2(MC_init_config_filename, MD_data_dir)
    #print('file write dir:', MD_data_dir, flush=True)


    for t in range(n_out_files):
        #print(q_list.shape, p_list.shape, l_list.shape, tau, save2file_strike, append_strike, gamma, temp)

        start_time = time.time()
        print('====== n out files ', t, flush=True)
        qpl_list = mdvv.nsteps(q_list, p_list, l_list, tau, save2file_strike, append_strike, gamma, temp)
        sec = time.time() - start_time
        print("nsamples {}, {} nitr --- {:.05f} sec ---".format(q_list.shape[0],save2file_strike,sec))
        #print("one forward step timing --- {:.05f} sec ---".format(sec/nitr))
        #quit()
        qpl_list = torch.stack(qpl_list, dim=2)  # shape [nsamples,3, traj_len, nparticles,dim]
        q_nxt = qpl_list[:, 0, -1, :, :]  # shape [nsamples,nparticles,dim]
        p_nxt = qpl_list[:, 1, -1, :, :]
   
        q_list = q_nxt
        p_list = p_nxt

        if t == 0:
            init_list = torch.unsqueeze(torch.stack((q_init, p_init, l_init), dim=1), dim=2)
            # shape [nsamples, 3, traj_len, nparticles, dim]
            qpl_list = torch.cat((init_list, qpl_list), dim=2)

        tmp_filename = MD_output_basenames + 'gamma{}_id'.format(gamma) + str(t) + '.pt'

        data2 = { 'qpl_trajectory':qpl_list, 'tau_short':tau_short, 'tau_long': tau_long }
        print('qpl traj ', qpl_list.shape, 'tau short', tau_short, 'tau_long' , tau_long, 'gamma', gamma, flush=True)
        torch.save(data2, tmp_filename)
 
#system_logs.print_end_logs()

