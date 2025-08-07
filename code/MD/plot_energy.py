import sys
sys.path.append( '../')

import itertools
import torch
import numpy as np
from utils.system_logs              import system_logs
from utils.mydevice                 import mydevice
import matplotlib.pyplot as plt


def de(e,npar):
    #shape = [trajectory, nsamples]
    e_shift = (e - e[0])/npar
    e_shift = e_shift.clone().detach().cpu().numpy()
    mean_e = np.mean(e_shift,axis=1)
    std_err_e = np.std(e_shift,axis=1) / np.sqrt(e.shape[1])
    return mean_e, std_err_e


if __name__ == '__main__':
    # python plot_e_conserve.py 128 0.3 0.46 1000 10 119 800000 lg

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
    if len(argv) != 6:
        print('usage <programe> <npar> <rho> <temp> <nsteps> <gamma> ' )
        quit()

    npar = int(argv[1])
    rho = argv[2]
    T = argv[3]
    nstep = int(argv[4])
    gamma = float(argv[5])

    if (gamma == 0.0) or (gamma == 1.0) or (gamma == 10.0) or (gamma == 20.0):
        print('gamma {} float to int .... '.format(gamma))
        gamma = int(gamma)

    data = {
	     # "energy1" : "../../../data_sets/gen_by_MD/3d/noML-metric-lt0.025every1t1000/n{}rho{}T{}/energy_gamma{}_tmax1000.pt".format(npar,rho,T, gamma),
        "energy2": "../../data_sets/gen_by_MD/3d/noML-metric-lt0.02every1t1000/n{}rho{}T{}/energy_gamma{}_tmax1000.pt".format(
            npar, rho, T, gamma),
        "energy3": "../../data_sets/gen_by_MD/3d/noML-metric-lt0.01every1t1000/n{}rho{}T{}/energy_gamma{}_tmax1000.pt".format(  npar, rho, T, gamma),
        "energy4": "../../data_sets/gen_by_MD/3d/noML-metric-lt0.01every1t1000/n{}rho{}T{}/energy_gamma{}_tmax1000.pt".format(
            npar, rho, T, gamma),
        # "energy4": "../../../data_sets/gen_by_MD/3d/noML-metric-lt0.001every1t1000/n{}rho{}T{}/energy_gamma{}_tmax1000.pt".format(
        #     npar, rho, T, gamma),
    }

    maindict = { "tau_long": 0.1,
                 "traj_len" : 8  }

    tau_long = maindict["tau_long"]
    traj_len = maindict["traj_len"]
    append_strike = 1
    input_seq_idx = traj_len - 1

    print('tau long', tau_long, 'input seq idx', input_seq_idx, 'nstep', nstep)

    # print('load md data file : ', data["energy1"]) # shape [trajectory, nsamples]
    print('load md data file : ', data["energy2"])
    print('load ml data file : ', data["energy3"])
    # data1 = torch.load(data["energy1"],map_location=map_location)
    # e1_append = data1["energy"][:nstep + 1]
    data2 = torch.load(data["energy2"],map_location=map_location)
    e2_append = data2["energy"][:nstep+1]
    data3 = torch.load(data["energy3"],map_location=map_location)
    e3_append = data3["energy"][:nstep+1]
    data4 = torch.load(data["energy4"],map_location=map_location)
    e4_append = data4["energy"][:nstep+1]

    #print(e1_append.shape, e2_append.shape, e3_append.shape)
    #print(torch.equal(e1_append[0], e2_append[0]))
    print(torch.equal(e2_append[0], e3_append[0]))

    #print(e1_append.shape, e2_append.shape, e3_append.shape)

    tau_traj_prep = tau_long * traj_len - tau_long
    t = np.arange(tau_traj_prep,nstep*tau_long + tau_traj_prep+tau_long ,tau_long*append_strike)

    # mean_e1, std_err_e1 = de(e1_append,npar)
    mean_e2, std_err_e2 = de(e2_append,npar)
    mean_e3, std_err_e3 = de(e3_append,npar)
    mean_e4, std_err_e4 = de(e4_append, npar)

    # print('saved energy shape', mean_e1.shape, std_err_e1.shape)
    print('saved energy shape', mean_e2.shape, std_err_e2.shape)
    print('saved energy shape', mean_e3.shape, std_err_e3.shape)
    print('saved energy shape', mean_e4.shape, std_err_e4.shape)

    #for i in range(len(mean_e1)):
    #  print('MD results', mean_e1[i], std_err_e1[i], mean_e2[i], std_err_e2[i])
    print(len(t),len(mean_e2))

    for i in range(len(mean_e3)):
      print('ML results', mean_e3[i], std_err_e3[i])

    # plt.title(r'n{}rho{}T{}$\gamma${}'.format(npar,rho,T,gamma),fontsize=20)
    # plt.xlabel(r'time' + '\n' + r'100000 iter (MD at $\tau=0.01$)' + '\n' + r'10000 iter (ML at $\tau=0.1$)', fontsize=18)
    plt.xlabel('time', fontsize=18)
    plt.ylabel(r'$\Delta E$', fontsize=18)
    # plt.errorbar(t, mean_e1, yerr=std_err_e1, errorevery=100, capsize=5, label=r'$\tau$=0.025')
    plt.errorbar(t,mean_e2, yerr=std_err_e2, errorevery=100, capsize=5, label=r'$\tau$=0.02')
    plt.errorbar(t,mean_e3, yerr=std_err_e3, errorevery=100, capsize=5, label=r'$\tau$=0.01')
    plt.errorbar(t,mean_e4, yerr=std_err_e4, errorevery=100, capsize=5, label=r'$\tau$=0.001')
    plt.tick_params(axis='x',   labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.grid()
    #plt.legend(loc='center right', fontsize=16)
    # plt.ylim(-2e-4,0.01)
    # plt.yscale('log')
    plt.ylim(-2e-4, 1)
    plt.legend(fontsize=16, loc='upper left')
    plt.tight_layout()
    plt.show()
