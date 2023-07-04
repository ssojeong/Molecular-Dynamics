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
        quit()

    load_files = argv[1]

    with open(load_files) as f:
        data = json.load(f)

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(34952)

    tau_cur = [0.01, 0.01,0.01]

    #nfiles = [data['c2_8'],data['c5_8'],data['c8_8']]
    #n       = [64, 64, 64]
    #rho = [0.053,0.3,0.72]
    #T   = [0.47, 0.46,0.48]
    #md = [1.57, 2.3, 2.4]

    #nfiles = [data['c1_8'],data['c4_8'],data['c7_8']]
    #n       = [32, 32, 32]
    #rho = [0.035,0.4,0.68]
    #T   = [0.46, 0.46,0.48]
    #md = [1.2, 2.37, 2.49]

    nfiles = [data['c3_8'],data['c6_8'],data['c9_8']]
    n       = [128, 128, 128]
    rho = [0.025,0.25,0.66]
    T   = [0.47, 0.44,0.47]
    md = [1.18, 2.3, 2.46]

    x       = [1,2,3]
    #marker = ['x', 'D', 'o']
    xx = [1.1,1.1,1.1]


    maindict = {
                "traj_len": 8,
                "level": 8,
	        "dt_samples" : 1, # no. of eps samples
                "tau_long" : 0.1} #101

    traj_len = maindict["traj_len"]
    input_seq_idx = traj_len - 1

    #if torch.cuda.is_available():
    #    map_location = lambda storage, loc: storage.cuda()
    #else:
    #    map_location = 'cpu'

    tau_long = maindict["tau_long"]

    fig, ax = plt.subplots(figsize=(5,6))

    count=0
    for i in nfiles:
      print('load file  ...... ', i , flush=True)

      avg_lambda=[]
      std_lambda=[]
      for j in range(5): 

        loadfile = torch.load(i +'_id{}.pt'.format(j),map_location=torch.device('cpu') )
        t_accum         = loadfile['t_accum'] #shape  [t_accum, avg dq]
        R_sample_append   = loadfile['R_sample_append']
        print('nsteps',t_accum, 'lambda',R_sample_append)
        R_sample_stack = torch.stack(R_sample_append,dim=1) #shape [nsamples, traj]
        LogR_sample_stack = torch.log2(R_sample_stack)
        LogR_sample_sum = torch.sum(LogR_sample_stack,dim=1) # shape [nsamples]
        #print('LogR_sample_sum shape', LogR_sample_sum)

        lambda_sample = 1. / (t_accum*tau_long) * LogR_sample_sum
        #print('lambda sample_sum ', lambda_sample)

        avg_lambda_np = np.mean(lambda_sample.detach().cpu().numpy())
        std_lambda_np = np.std(lambda_sample.detach().cpu().numpy()) / math.sqrt(lambda_sample.shape[0])
        print('id{}.pt'.format(j), 'numpy : avg lambda {:.3f}'.format(avg_lambda_np), 'std lambda {:.3f}'.format(std_lambda_np))
        avg_lambda.append(avg_lambda_np)
        std_lambda.append(std_lambda_np)
 
      avg_lambda_sample_np = np.mean(avg_lambda)
      std_lambda_sample_np = np.mean(std_lambda)
      print('all samples ', 'numpy : avg lambda {:.3f}'.format(avg_lambda_sample_np), 'std lambda {:.3f}'.format(std_lambda_sample_np))

      if count == 0 :
           plt.title('n={} particles, window-sliding={}'.format(n[count],maindict["level"]),fontsize=20)
           plt.plot(x[count], md[count], marker="*", markersize=10, color='black', zorder=3,linestyle = 'None',
                    label=r'VV $\tau={}$'.format(tau_cur[count]))
           plt.errorbar(x[count], avg_lambda_sample_np, marker='D',color='k', yerr= std_lambda_sample_np,capsize=7, mfc='none', capthick=2,label=r'LLUF $\tau={}$'.format(maindict["tau_long"]))
      else:
           plt.plot(x[count], md[count], marker="*", markersize=10, color='black', zorder=3,linestyle = 'None')
           plt.errorbar(x[count], avg_lambda_sample_np, marker='D',color='k', yerr= std_lambda_sample_np,capsize=7, mfc='none', capthick=2)

      plt.xlabel(r'$\rho$',fontsize=20)
      plt.ylabel(r'$\lambda$',fontsize=20)
      plt.legend(loc='lower right', fontsize=12)
      count += 1

    plt.xticks(x,rho, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    fig_filename = 'analysis/lyapunov/n{}level{}.pdf'.format(n[0],maindict["level"])
    plt.show()
    #fig.savefig(fig_filename,bbox_inches='tight', dpi=200)

system_logs.print_end_logs()

