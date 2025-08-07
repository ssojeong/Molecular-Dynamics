import sys 
sys.path.append( '../../')

import torch

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # python plot_rdf.py 128 0.3 0.46 0.1 10000 10 119 800000 lg

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    argv = sys.argv
    if len(argv) != 9:
        print('usage <programe> <npar> <rho> <temp> <nstep> <gamma> <saved model> <dpt> <region>' )
        quit()

    npar = int(argv[1])
    rho = argv[2]
    dim = int(argv[3])
    T = float(argv[4])
    gamma = argv[5]
    saved_model = argv[6]
    dpt = int(argv[7])
    region = argv[8]

    data = { 
            "filename1" : '../../../data_sets/gen_by_MD/{}d/noML-metric-lt0.01every1t0.7t1000/n{}rho{}T{}/'.format(dim,npar,rho,T)
             + 'rij_gr_gamma{}.pt'.format(gamma),
            "filename2": '../../../data_sets/gen_by_ML/lt0.1dpt{}_{}/n{}rho{}T{}/'.format(dpt,region,npar,rho,T)
                         + 'rij_gr_gamma{}mb{}.pt'.format(gamma, saved_model),
            "filename3": '../../../data_sets/gen_by_ML/lt0.1dpt100000_{}/n{}rho{}T{}/'.format(region, npar, rho, T)
                     + 'rij_gr_gamma{}mb009.pt'.format(gamma),
            "nofile": 2,
            "linestyle" : ['dashed','dotted'],
            }

    d = {}
    for i in range(data["nofile"]):
        print('load file', data, flush=True)
        d["data" + str(i)] = torch.load(data["filename{}".format(i + 1)], map_location=map_location)


    dd = {}
    for i in range(data["nofile"]):
        dd["rmid" + str(i)] = d["data" + str(i)]['nrmid'] # save nrmid over time steps
        dd["gr" + str(i)] = d["data" + str(i)]['ngr'] # save ngr over time steps
        print('saved time points ',len(dd["rmid" + str(i)]),len(dd["gr" + str(i)]), flush=True)

    plt.figure()

    nrmid_mc = dd["rmid0"][0] # take rmid at initial time
    ngr_mc = dd["gr0"][0] # take gr at initial time

    nrmid_vv = dd["rmid0"][-1] # take rmid at last time
    ngr_vv = dd["gr0"][-1] # take gr at last time

    nrmid_lluf = dd["rmid1"][-1] # take rmid at last time
    ngr_lluf = dd["gr1"][-1] # take gr at last time

    ngr_mc = [tensor.cpu() for tensor in ngr_mc]
    ngr_vv = [tensor.cpu() for tensor in ngr_vv]
    ngr_lluf = [tensor.cpu() for tensor in ngr_lluf]

    print('mc gr', len(ngr_mc), ngr_mc[0])
    print('vv gr', len(ngr_vv), ngr_vv[0])
    print('ml gr', len(ngr_lluf), ngr_lluf[0])

    print(np.array(nrmid_mc).shape, np.array(ngr_mc).shape)
    print(np.array(nrmid_vv).shape, np.array(ngr_vv).shape)
    print(np.array(nrmid_lluf).shape, np.array(ngr_lluf).shape)

    plt.plot(np.array(nrmid_mc), np.array(ngr_mc), color='k', mfc='none')#, label='MC')
    plt.plot(np.array(nrmid_vv), np.array(ngr_vv), color='k', mfc='none', linestyle='dashed')#, label='VV')
    plt.plot(np.array(nrmid_lluf), np.array(ngr_lluf), color='k', mfc='none', linestyle='dotted')#, label=f'LLUF(n={npar})')

    if  (npar == 64 or npar == 128) and region == 'lg' and (T==0.46 or T==0.48):
        print('use trained model from n=64 or 128 ....')

        data_n128 = torch.load(data["filename3"])
        nrmid_n128 = data_n128["nrmid"]
        ngr_n128 = data_n128['ngr']
        print('saved time points ', len(nrmid_n128), len(ngr_n128), flush=True)

        nrmid_lluf128 = nrmid_n128[-1]
        ngr_lluf128 = ngr_n128[-1]
        # ngr_lluf128 = [tensor.cpu() for tensor in ngr_lluf128]

        print(np.array(nrmid_lluf128).shape, np.array(ngr_lluf128).shape)
        plt.plot(np.array(nrmid_lluf128), np.array(ngr_lluf128), color='r', mfc='none', linestyle='dotted')#, label=f'LLUF(n={npar})')

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(0, 3)

    if region == 'g':
        plt.ylim(0, 11)
    if region == 'lg':
        plt.ylim(0, 7)
    if region == 'l':
        plt.ylim(0, 4)

    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.grid()
    plt.savefig( f'../../analysis/{dim}d/figures/rdf/npar{npar}rho{rho}gamma{gamma}nstep10000_rdf_T{T}.pdf', bbox_inches='tight', dpi=200)
    plt.close()
   
