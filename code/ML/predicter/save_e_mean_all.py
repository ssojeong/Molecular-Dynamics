import sys
sys.path.append( '../../')

import itertools
import torch
import numpy as np
from utils.system_logs              import system_logs
from utils.mydevice                 import mydevice
import matplotlib.pyplot as plt
import os

def de(e,npar):
    e = e.clone().detach().cpu().numpy()
    #shape = [trajectory, nsamples]
    mean_e = np.mean(e, axis=1)
    mean_e_shift = abs(mean_e - mean_e[0])/npar
    return mean_e_shift[-1]

if __name__ == '__main__':
    # python plot_e_conserve.py 128 0.035 0.49  1000 1

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
        print('usage <programe> <npar> <rho> <dim> <gamma> <saved_model> <dpt> <region> <n split> ' )
        quit()

    npar = int(argv[1])
    rho = argv[2]
    dim = int(argv[3])
    gamma = float(argv[4])
    saved_model = argv[5]
    dpt = int(argv[6])
    region = argv[7]
    n_split = int(argv[8])

    if (gamma == 0.0) or (gamma == 1.0) or (gamma == 10.0) or (gamma == 20.0):
        print('gamma {} float to int .... '.format(gamma))
        gamma = int(gamma)

    data0_list = []
    data1_list = []
    data2_list = []
    temp_list = [0.44, 0.46, 0.48, 0.5]

    for i in temp_list:
        data = {
            "energy1": "../../../data_sets/gen_by_ML/lt0.1dpt100000_{}/n{}rho{}T{}/energy_gamma{}mb009_nsteps10000.pt".format(region, npar, rho, i, gamma),
             "energy2" : "../../../data_sets/gen_by_MD/{}d/noML-metric-lt0.01every1t0.7t1000/n{}rho{}T{}/energy_gamma{}_tmax1000.pt".format(dim, npar,rho,i,gamma),
             "energy3" : "../../../data_sets/gen_by_ML/lt0.1dpt{}_{}/n{}rho{}T{}/energy_gamma{}mb{}_nsteps10000.pt".format(dpt,region,npar,rho,i,gamma,saved_model)}

        data1_list.append(data['energy2'])
        data2_list.append(data['energy3'])

        if (npar == 64 or npar ==128) and region == 'lg' : #and i != 0.5:
            print('filename0 is ',i)
            data0_list.append(data['energy1'])


    nsamples = 1000 # test set

    if (npar == 64 or npar == 128) and region == 'lg':

        e0 = {}
        for i in range(len(data0_list)):
            print('load file', i, flush=True)
            data0 = torch.load(data0_list[i], map_location=map_location)
            e0["data0_" + str(i)] = data0["energy"] # shape [trajectory, nsamples]
            print(e0["data0_" + str(i)].shape)

        emean0 = {}
        emean0_append = []
        for i in range(len(data0_list)):
            for j in range(0, nsamples, n_split):
                print('samples', j, '....', j + n_split, 'T', i)
                emean0["data0_" + str(i)] = de(e0["data0_" + str(i)][:,j:j+n_split], npar) # shape [] take last time point
                emean0_append.append(emean0["data0_" + str(i)])

        e_mu0 = []
        e_se0 = []
        for k in range(len(temp_list)):
            e_mu_mean0 = np.mean(
                emean0_append[int(nsamples / n_split) * k : int(nsamples / n_split) + int(nsamples / n_split) * k])
            e_mu_std0 = np.std(
                emean0_append[int(nsamples / n_split) * k : int(nsamples / n_split) + int(nsamples / n_split) * k]) / np.sqrt(
                len(emean0_append[int(nsamples / n_split) * k : int(nsamples / n_split) + int(nsamples / n_split) * k]))
            e_mu0.append(e_mu_mean0)
            e_se0.append(e_mu_std0)
            print(k, e_mu_mean0, e_mu_std0)

    e1 = {}
    for i in range(len(data1_list)):
        print('load file', i, flush=True)
        data1 = torch.load(data1_list[i],map_location=map_location)
        e1["data1_" + str(i)] = data1["energy"] # shape [trajectory, nsamples]
        print(e1["data1_" + str(i)].shape)

    e2 = {}
    for i in range(len(data2_list)):
        print('load file', i, flush=True)
        data2 = torch.load(data2_list[i],map_location=map_location)
        e2["data2_" + str(i)] = data2["energy"] # shape [trajectory, nsamples]
        print(e2["data2_" + str(i)].shape)

    e_mean1 = {}
    e_mean2 = {}
    e_mean1_append = []
    e_mean2_append = []

    for i in range(len(data1_list)): # over temperature
        for j in range(0,nsamples,n_split): # all samples to 5 groups
            print('samples', j, '....', j + n_split)
            e_mean1["data1_" + str(i)] = de(e1["data1_" + str(i)][:,j:j+n_split], npar) # shape [] ;take last time point
            e_mean2["data2_" + str(i)] = de(e2["data2_" + str(i)][:,j:j+n_split], npar) # shape [] ;take last time point
            e_mean1_append.append(e_mean1["data1_" + str(i)])
            e_mean2_append.append(e_mean2["data2_" + str(i)])


    e_mu1 = []
    e_mu2 = []
    e_se1 = []
    e_se2 = []

    for k in range(len(temp_list)):

        e_mu_mean1 = np.mean(e_mean1_append[int(nsamples/n_split) * k : int(nsamples/n_split) + int(nsamples/n_split) * k])
        e_mu_std1 = np.std(e_mean1_append[int(nsamples/n_split) * k : int(nsamples/n_split) + int(nsamples/n_split) * k]) / np.sqrt(
                    len(e_mean1_append[int(nsamples/n_split)*k:int(nsamples/n_split)+int(nsamples/n_split)*k]))

        # print('cv mean2',cv_mean2_append[int(1000/n_split)*k:int(1000/n_split)+int(1000/n_split)*k])
        e_mu_mean2 = np.mean(e_mean2_append[int(nsamples/n_split) * k : int(nsamples/n_split)+int(nsamples/n_split) * k])
        e_mu_std2 = np.std(e_mean2_append[int(nsamples/n_split) * k :
                    int(nsamples/n_split)+int(nsamples/n_split)*k]) / np.sqrt(
                    len(e_mean2_append[int(nsamples/n_split)*k:int(nsamples/n_split)+int(nsamples/n_split)*k]))

        e_mu1.append(e_mu_mean1)
        e_mu2.append(e_mu_mean2)
        e_se1.append(e_mu_std1)
        e_se2.append(e_mu_std2)
        print(k, e_mu_mean1, e_mu_std1, e_mu_mean2, e_mu_std2)

    fig, axes = plt.subplots(1, 1, figsize=(6, 5))

    # plt.suptitle(r'npar={} $\rho$={} $\gamma$={}'.format(npar, rho, gamma)  , fontsize=15)

    saved_txtfile = f'../../analysis/{dim}d/npar{npar}rho{rho}gamma{gamma}_e.txt'

    # Check if the file exists, and remove it if it does
    if os.path.exists(saved_txtfile):
        os.remove(saved_txtfile)
        print(f"{saved_txtfile} was removed.")

    for j in np.arange(len(data1_list)):
        # ax.set_title(title[j])
        if (npar == 64 or npar == 128) and region == 'lg' :# and j < len(data0_list):
            vv_mean = e_mu1[j]
            vv_se = e_se1[j]
            ml_mean = e_mu2[j]
            ml_se = e_se2[j]
            ml128_mean = e_mu0[j] # 64 or 128
            ml128_se = e_se0[j] # 64 or 128

            with open(saved_txtfile, 'a') as file:
                print('save file end of time point....')
                print('vv mean', vv_mean, 'se', vv_se,
                      'ml mean', ml_mean, 'se',ml_se ,
                      'ml128 mean', ml128_mean, 'se',ml128_se)
                file.write(f'{j} {vv_mean} {vv_se} {ml_mean} {ml_se} {ml128_mean} {ml128_se}\n')

        else:
            vv_mean = e_mu1[j]
            vv_se = e_se1[j]
            ml_mean = e_mu2[j]
            ml_se = e_se2[j]

            with open(saved_txtfile, 'a') as file:
                print('save file end of time point....')
                print('vv mean', vv_mean, 'se', vv_se,
                      'ml mean', ml_mean, 'se',ml_se)
                file.write(f'{j} {vv_mean} {vv_se} {ml_mean} {ml_se}\n')

    print(len(data0_list), len(data1_list), len(data2_list))
