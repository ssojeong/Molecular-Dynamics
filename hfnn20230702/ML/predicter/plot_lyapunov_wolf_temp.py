import sys 
sys.path.append('../../')
from utils.system_logs              import system_logs
from utils.mydevice                 import mydevice
import matplotlib.pyplot as plt
import torch
import numpy as np
import json
import math


if __name__ == '__main__':

    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    argv = sys.argv
    if len(argv) != 2:
        print('usage <programe> <json file>')
        #print('usage <programe> <npar> <rho> <temp> ' )
        quit()

    load_files = argv[1]

    with open(load_files) as f:
        data = json.load(f)

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(34952)

    nfiles = [data['test1'],data['test2'],data['test3'],data['test4'],data['test5'],data['test6']]
    tau_cur = [0.01, 0.001, 0.01, 0.001, 0.01, 0.001]
    n       = [32, 32, 128, 128,  64, 64]
    x       = [1,1, 2, 2, 3, 3]
    rho = [0.6,0.6,0.6,0.6, 0.6, 0.6]
    T   = [0.6, 0.6,0.7,0.7,1.2,1.2]
    marker = ['x', 'D','x', 'D','x', 'D']
    xx = [1.1,1.1, 1.1, 1.1, 3.1, 3.1]
    ref = [2.62,2.62, 2.62 , 2.62, 3.36, 3.36]

    maindict = {
                "traj_len": 8,
	        "dt_samples" : 1, # no. of eps samples
                "t_max" : 51} #101

    t_max = maindict["t_max"]
    traj_len = maindict["traj_len"]
    input_seq_idx = traj_len - 1

    #if torch.cuda.is_available():
    #    map_location = lambda storage, loc: storage.cuda()
    #else:
    #    map_location = 'cpu'

    t_max = maindict["t_max"]

    fig, ax = plt.subplots(figsize=(6,6))

    count=0
    for i in nfiles:
        print('load file  ...... ', i, flush=True)
        loadfile = torch.load(i,map_location=torch.device('cpu') )
        t_accum         = loadfile['t_accum'] #shape  [t_accum, avg dq]
        t_accum = torch.Tensor(t_accum)
        R_sample_append   = loadfile['R_sample_append']
        R_sample_stack = torch.stack(R_sample_append,dim=1) #shape [nsamples, traj]
        choose_t_accum =  t_accum[t_accum >= round(t_max / tau_cur[count])]
        choose_t_accum = choose_t_accum[0].to(torch.int).item()
        idx = (t_accum == choose_t_accum).nonzero().item()
        LogR_sample_stack = torch.log2(R_sample_stack[:,:idx+1])
        LogR_sample_sum = torch.sum(LogR_sample_stack,dim=1) # shape [nsamples]

        lambda_sample = 1. / t_max * LogR_sample_sum
        #print('lambda sample_sum ', lambda_sample)

        avg_lambda_sample_np = np.mean(lambda_sample.detach().cpu().numpy())
        std_lambda_sample_np = np.std(lambda_sample.detach().cpu().numpy()) / math.sqrt(lambda_sample.shape[0])
        print('tau cur ', tau_cur[count], 'nsteps ', choose_t_accum, 'numpy : avg lambda', avg_lambda_sample_np, 'std lambda', std_lambda_sample_np)
        plt.plot(xx[count],ref[count],marker="*", markersize=10, color='black',zorder=3)
        if count == 0 or count ==1:
            plt.errorbar(x[count], avg_lambda_sample_np, marker=marker[count],color='k', yerr= std_lambda_sample_np,capsize=7, mfc='none', capthick=2,
                     label=r'$\rho={}, \tau={}$'.format(rho[count],tau_cur[count]))
        else:
            plt.errorbar(x[count], avg_lambda_sample_np, marker=marker[count],color='k', yerr= std_lambda_sample_np,capsize=7,mfc='none', capthick=2)

        plt.xlabel(r'$T$',fontsize=20)
        plt.ylabel(r'$\lambda$',fontsize=20)
        plt.legend(loc='upper left', fontsize=20)
        count += 1

    plt.xticks(x,T, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()
  
system_logs.print_end_logs()

