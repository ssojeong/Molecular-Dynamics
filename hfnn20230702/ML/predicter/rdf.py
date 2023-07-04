import sys 
sys.path.append( '../../')

import torch
from utils.system_logs              import system_logs
from utils.mydevice                 import mydevice
import math
import matplotlib.pyplot as plt 
from utils.pbc                         import pairwise_dq_pbc
from utils.get_paired_distance_indices import get_paired_distance_indices
from collections import Counter

def pack_data(qpl_list):

    # shape = [nsamples, (q,p,boxsize), trajetory,  nparticles, DIM]
    q_traj = qpl_list[:,0,:,:,:].clone().detach()
    p_traj = qpl_list[:,1,:,:,:].clone().detach()
    l_init = qpl_list[:,2,:,:,:].clone().detach()

    return q_traj,p_traj,l_init

def l_max_distance(l_list):
    boxsize = torch.mean(l_list)
    L_h = boxsize / 2.
    q_max = math.sqrt(L_h * L_h + L_h * L_h)
    print('boxsize', boxsize.item(), 'maximum distance dq = {:.2f}, dq^2 = {:.2f}'.format(q_max, q_max * q_max))
    return boxsize, q_max

def pairDistributionFunction(grdelta,grbinmax,q_list,p_list,l_list):
    """Generates a pair-distribution function on given data"""

    # q_list shape [nsample, nparticle, dim]
    nsamples = q_list.shape[0]
    nparticles = q_list.shape[1]
    dim = q_list.shape[2]
    boxsize = torch.mean(l_list)
    halfboxsize = boxsize/2.
    print("Generating rdf..."),

    dq  = pairwise_dq_pbc(q_list,l_list) #shape = [nsamples,nparticles,nparticles,dim]
    idx = get_paired_distance_indices.get_indices(dq.shape) # offset i particle, i particle
    dr  = get_paired_distance_indices.reduce(dq,idx)
    dr  = dr.view([nsamples,nparticles,nparticles-1,dim])
    
    r = torch.sqrt(torch.sum(dr*dr,dim=-1))
    # r shape [nsample, nparticle, nparticle-1]
    r = r.view(nsamples,-1) 
    # r shape [nsample, nparticle*nparticle-1]
    #print('grdelta',  grdelta, 'half boxsize', halfboxsize.item())
    assert((r < math.sqrt(halfboxsize*halfboxsize + halfboxsize*halfboxsize)).all() == True), 'rij go out box'

    #indices = torch.where(r < halfboxsize) 
    grbin = (r/grdelta).to(torch.long) # bin corresponding to the interval (r,r+dr)
    indices = torch.where(grbin < grbinmax)
    #print('grbin smaller than grbinmax', grbin)
    #print('count', Counter(grbin[indices].tolist()))
    #print(grbin[indices].tolist())
    count = Counter(grbin[indices].tolist())
    sorted_key = list(dict(sorted(count.items())))
    sorted_value = list(dict(sorted(count.items())).values())
    mean_sorted_value = [hist/nsamples for hist in sorted_value]
    #print('sort key', sorted_key)
    #print('sort value', sorted_value)
    #print('mean sort value', mean_sorted_value)

    return sorted_key, mean_sorted_value

if __name__ == '__main__':

    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    argv = sys.argv
    if len(argv) != 8:
        print('usage <programe> <npar> <rho> <temp> <level> <tau max> <phase> <name>' )
        quit()

    param1 = argv[1]
    param2 = argv[2]
    param3 = argv[3]
    param4 = int(argv[4])
    param5 = float(argv[5])
    param6 = argv[6]
    param7 = argv[7]

    states = { "npar" : param1,
               "rho"  : param2,
               "T"    : param3,
               "level": param4,
               "tau_max" : param5,
               "phase": param6,
               "name" : param7
              }

    npar = states["npar"]
    rho = states["rho"]
    T = states["T"]
    level = states["level"]
    tau_max =states["tau_max"]
    name = states["name"]
    phase = states["phase"]
 
    data = { 
             "filename1" : '../../../data_sets/gen_by_MD/noML-metric-lt0.01every0.1t0.7t100/n{}rho{}T{}/'.format(npar,rho,T)
             + 'n{}rho{}T{}.pt'.format(npar,rho,T),
             #"filename2": '../../../data_sets/gen_by_MD/noML-metric-lt0.001every0.1t0.7t100/n{}rho{}T{}/'.format( npar, rho,T)
             #+ 'n{}rho{}T{}.pt'.format(npar, rho, T) ,
             "filename2": '../../../data_sets/gen_by_ML/lt0.1dpt1800000/n{}rho{}T{}/'.format(npar,rho,T) +"{}".format(states["name"]),
            "nofile": 2,
            "mode" : ['MD','ML'],
            #"label": [r'VV $\tau$=1e-2', r'VV $\tau$=1e-3'],
           "label": [r'VV $\tau$=1e-2', r'LLUF level={} $\tau$=0.1'.format(level)],
           "linestyle" : ['--','dotted']
            }

    maindict = {"tau_max"  : tau_max,
                "tau_long" : 0.1,
                "grdelta" : 0.06,
                "traj_len" : 8, 
                "everytau" : 0.1, # md = 0.1 at tau=0.1 or 0.2 at tau=0.2  ; ml = 0.1
                "everysave" : 0.1,
                "therm_state": 'n{}rho{}T{}_{}'.format(npar, rho, T, phase)
                }

    d = {}
    for i in range(data["nofile"]):
        print('load file', data, flush=True)
        d["data" + str(i)] = torch.load(data["filename{}".format(i + 1)], map_location=map_location)

    traj_len = maindict["traj_len"]
    everysave = maindict["everysave"]
    everytau = maindict["everytau"]
    input_seq_idx = traj_len - 1
    nstep = round( (maindict["tau_max"] - (input_seq_idx * maindict["tau_long"] )) / maindict["tau_long"])
    therm_state = maindict["therm_state"]
    pair_step_idx = round(everytau/everysave)
    grdelta = maindict["grdelta"]

    dd = {}
    for i in range(data["nofile"]):

        dd["qpl_traj" + str(i)] = d["data" + str(i)]['qpl_trajectory'][:,:,0:nstep+1:pair_step_idx,:,:]
        # shape = [nsamples, (q,p,boxsize), trajetory,  nparticles, DIM]
        print('qpl traj shape',dd["qpl_traj" + str(i)].shape, 'level ', level, 'nsteps ',nstep, flush=True)

        dd["q_traj"+ str(i)], dd["p_traj"+ str(i)], l_list = pack_data(dd["qpl_traj" + str(i)])
        print('shape', dd["q_traj0"].shape, dd["p_traj0"].shape, l_list.shape, flush=True)
        # shape = [nsamples,  trajectory,  nparticles, DIM]
        boxsize, q_max = l_max_distance(l_list)

    _, _, traj_len, nparticles, _ = dd["qpl_traj0"].shape

    grbinmax = int((boxsize/2)/grdelta)
    print('boxsize', boxsize.item(), 'grdelta', grdelta,'grbinmax',grbinmax, 'nsteps', nstep)

    plt.figure()
    time = [0, nstep]
    #time = np.arange(input_seq_idx*everytau,qpl_traj.shape[2]*everytau,everytau*n_chain) #* tau_long

    r_ij = []
    count = 0
    for i in range(data["nofile"]):

        for step in time:

           sorted_grbin, mean_sorted_value = pairDistributionFunction(grdelta, grbinmax, dd["q_traj"+str(i)][:,step], dd["p_traj"+str(i)][:,step], l_list[:,step])

           nrmid=[]
           ngr=[]
           # normalize gr
           for grbin, value in zip(sorted_grbin, mean_sorted_value):
              rinner = grbin * grdelta
              router = rinner + grdelta
              shellvol = 2 * math.pi * rinner * grdelta  #math.pi * (router**2 - rinner**2)
              gr =  (boxsize**2 /nparticles) / (nparticles-1)  * value  / shellvol
              rmid = rinner + 0.5*grdelta
              r_ij.append(rmid)
              nrmid.append(rmid)
              ngr.append(gr)

           print('count ', count, ' len gr', len(ngr),'sum gr', sum(ngr))

           if count ==0: # start time
               print('start ======= ', step)
               plt.plot(nrmid,ngr, color='k', label='t={:.2f}'.format(input_seq_idx*everysave))
                   #mfc='none', linestyle= (0,(1,10)),label='t={:.2f}'.format(input_seq_idx*everysave))
               plt.xlabel(r'$r$',fontsize=20)
               plt.ylabel(r'$g(r)$',fontsize=20)

           elif count != 0 and step ==0:
               print("count ", count, 't=',step, '....pass....')
               pass

           else:
               print('nsteps ======= ', step)
               print(' plot level {} time .....{}'.format(level, input_seq_idx*everysave + step*everytau))
               plt.plot(nrmid,ngr, color='k', mfc='none', linestyle= data["linestyle"][count//len(time)],
                       label='t={:.2f} {}'.format(input_seq_idx*everysave + (step)*everytau,data["label"][count//len(time)]))

           count += 1

        print('min r', min(r_ij), 'max r', max(r_ij))
        plt.title(therm_state + '\n' + r'min r {:.3f}  max r {:.3f}'.format(min(r_ij),max(r_ij)),fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim(0, 3)
        plt.legend(loc="upper right", fontsize=12)
        plt.tight_layout()
        plt.grid()

    #plt.show()
    fig_filename = 'analysis/rdf/' + therm_state + '_level{}_{}step.pdf'.format(level,nstep)
    plt.savefig(fig_filename, bbox_inches='tight', dpi=200)
    plt.close()
   
