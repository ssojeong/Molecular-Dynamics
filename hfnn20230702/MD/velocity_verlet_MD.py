import sys
sys.path.append( '../')

from utils.pbc import pbc
from utils.utils import assert_nan
from hamiltonian.lennard_jones2d import lennard_jones2d

from utils.system_logs           import system_logs
from utils.mydevice              import mydevice
from utils.force_MD import force

import numpy as np
import shutil
import json
import time
import torch

class velocity_verlet_MD:

    def __init__(self,lennard_jones2d):
        self.lennard_jones2d = lennard_jones2d
        print(' velocity verlet MD')
        self.f_stat = force()
        print(' check force .....')

        #assert(self.modef == 'ff'),'hf mode not implemented in velocity_verlet3'

    def one_step(self,q_list,p_list,l_list,tau):

        p_list_2 = p_list - 0.5*tau*self.lennard_jones2d.derivative(q_list,l_list)
        f1 = -self.lennard_jones2d.derivative(q_list, l_list)
        #self.f_stat.accumulate(1, f1, tau)
        #self.f_stat.print()

        q_list_2 = q_list + tau*p_list_2
        q_list_new = pbc(q_list_2,l_list)

        p_list_new = p_list_2 - 0.5*tau*self.lennard_jones2d.derivative(q_list_new,l_list)
        f2 = -self.lennard_jones2d.derivative(q_list_new, l_list)
        #self.f_stat.accumulate(2, f2, tau)
        #self.f_stat.print()

        assert_nan(p_list_new)
        assert_nan(q_list_new)
        return q_list_new,p_list_new,l_list


    def nsteps(self,q_list,p_list,l_list,tau,nitr,append_strike):

        assert (nitr % append_strike == 0), 'incompatible strike and nitr'

        qpl_list = []

        for t in range(nitr):
            #print('====== step ', t, flush=True)
            q_list,p_list,l_list = self.one_step(q_list,p_list,l_list,tau)
            nxt_qpl = torch.stack((q_list, p_list, l_list), dim=1)
  
            if t%500 == 0: print('.',end='',flush=True)

            if (t + 1) % append_strike == 0:
                print('====== step ', t+1, flush=True)
                qpl_list.append(nxt_qpl)
            # print('cpu used:', psutil.cpu_percent(), '\n')
            # print('memory % used:', psutil.virtual_memory()[2], '\n')

        return qpl_list

if __name__ == '__main__':

    argv = sys.argv
    MDjson_file = argv[1]

    with open(MDjson_file) as f:
        data = json.load(f)

    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()  

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(34952)

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
    mdvv = velocity_verlet_MD(potential_function)

    q_list = q_init.clone().detach()
    p_list = p_init.clone().detach()
    l_list = l_init.clone().detach()

    shutil.copy2(MC_init_config_filename, MD_data_dir) 
    print('file write dir:', MD_data_dir, flush=True)


    for t in range(n_out_files):

        start_time = time.time()
        print('====== n out files ', t, flush=True)
        qpl_list = mdvv.nsteps(q_list, p_list, l_list, tau, save2file_strike, append_strike)
        sec = time.time() - start_time
        print("nsamples {}, {} nitr --- {:.05f} sec ---".format(q_list.shape[0],save2file_strike,sec))
        #print("one forward step timing --- {:.05f} sec ---".format(sec/nitr))

        qpl_list = torch.stack(qpl_list, dim=2)  # shape [nsamples,3, traj_len, nparticles,dim]
        q_nxt = qpl_list[:, 0, -1, :, :]  # shape [nsamples,nparticles,dim]
        p_nxt = qpl_list[:, 1, -1, :, :]
   
        q_list = q_nxt
        p_list = p_nxt

        tmp_filename = MD_output_basenames + '_id' + str(t) + '.pt'
        data2 = { 'qpl_trajectory':qpl_list, 'tau_short':tau_short, 'tau_long': tau_long } 
        print('qpl traj ', qpl_list.shape, 'tau short', tau_short, 'tau_long' , tau_long, flush=True)
        torch.save(data2, tmp_filename)
 
#system_logs.print_end_logs()

