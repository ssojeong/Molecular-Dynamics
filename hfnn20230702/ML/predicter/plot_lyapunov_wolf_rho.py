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

    tau_cur = [0.01, 0.001, 0.01, 0.001, 0.01, 0.001]

    nfiles = [data['c1_1'],data['c1_2'],data['c4_1'],data['c4_2'],data['c7_1'],data['c7_2']]
    n       = [32, 32, 32, 32,  32, 32]
    rho = [0.035,0.035,0.4,0.4, 0.68, 0.68]
    T   = [0.47, 0.47,0.46,0.46,0.47,0.47]

    #nfiles =  [data['c2_1'],data['c2_2'],data['c5_1'],data['c5_2'],data['c8_1'],data['c8_2']]
    #n       = [64, 64, 64, 64,  64, 64]
    #rho = [0.053 ,0.053, 0.3, 0.3, 0.72,0.72]
    #T   =  [0.47,0.47, 0.45,0.45, 0.48, 0.48] 

    #nfiles =  [data['c3_1'],data['c3_2'],data['c6_1'],data['c6_2'],data['c9_1'],data['c9_2']]
    #n       = [128, 128, 128, 128,  128, 128]
    #rho = [0.025,0.025, 0.25, 0.25, 0.66, 0.66]
    #T   =  [0.47,0.47,  0.44,0.44,0.47,0.47]

    x       = [1,1, 2, 2, 3, 3]
    marker = ['x', 'D','x', 'D','x', 'D']
    xx = [1.1,1.1, 1.1, 1.1, 3.1, 3.1]

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

    fig, ax = plt.subplots(figsize=(5,6))

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
        #print('LogR_sample_sum shape', LogR_sample_sum)

        lambda_sample = 1. / t_max * LogR_sample_sum
        #print('lambda sample_sum ', lambda_sample)

        avg_lambda_sample_np = np.mean(lambda_sample.detach().cpu().numpy())
        std_lambda_sample_np = np.std(lambda_sample.detach().cpu().numpy()) / math.sqrt(lambda_sample.shape[0])
        print('tau cur ', tau_cur[count], 'nsteps ', choose_t_accum, 'numpy : avg lambda', avg_lambda_sample_np, 'std lambda', std_lambda_sample_np)
        if count == 0 or count ==1:
           plt.title('n={} particles'.format(n[count]),fontsize=20)
           plt.errorbar(x[count], avg_lambda_sample_np, marker=marker[count],color='k', yerr= std_lambda_sample_np,capsize=7, mfc='none', capthick=2,
                     label=r'$\tau={}$'.format(tau_cur[count]))
        else:
           plt.errorbar(x[count], avg_lambda_sample_np, marker=marker[count],color='k', yerr= std_lambda_sample_np,capsize=7, mfc='none', capthick=2)

        plt.xlabel(r'$\rho$',fontsize=20)
        plt.ylabel(r'$\lambda$',fontsize=20)
        plt.legend(loc='center right', fontsize=20)
        count += 1

    plt.xticks(x,rho, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()

system_logs.print_end_logs()

