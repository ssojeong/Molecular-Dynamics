import sys
import json
import torch
import itertools
import matplotlib.pyplot as plt

def crash_accum(n_file, time_step):

    crash_info = torch.load(n_file)
    crash_iter = crash_info['crash_niter']
    crash_ct = crash_info['crash_nct']
    
    crash_ts = crash_iter * time_step  + time_step 
    accum = itertools.accumulate(crash_ct)

    print(crash_iter.shape, crash_ts.shape, crash_ct.shape, torch.sum(crash_ct))

    return crash_ts, accum

if __name__ == '__main__':

    argv = sys.argv
    json_file = argv[1]	
    
    with open(json_file) as f:
        data = json.load(f)

    file_list = data['file_list']
    filenames = data['filenames']
    file_ts   = data['file_ts']
    color     = data['color']
   
    #print(file_list)
    
    giter={}
    gaccum={}
    gname={}

    for i in range(len(file_list)):
        giter["crash_iter" + str(i+1)], accum = crash_accum(file_list[i], file_ts[i])
        gaccum["accum" + str(i+1)] = list(accum)

    for i in range(len(file_list)):
       plt.title('accumulate function of time at crash',fontsize=10)
       plt.plot(giter["crash_iter" + str(i+1)], gaccum["accum" + str(i+1)], color[i], alpha=.5, label = filenames[i])
       plt.xlabel('time',fontsize=12)
       plt.ylabel('accumulate',fontsize=12)
       plt.tick_params(labelsize=10)
       #plt.legend(loc='lower right', columnspacing=1, fontsize='xx-small', handlelength=1.8, labelspacing=0.01)
       plt.legend(loc='lower right', columnspacing=1, handlelength=1.8, labelspacing=0.01)
    plt.grid()
    plt.show()

