import torch
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob

def Specific_heat(pe, npar,T):

    # pe shape [trajectory,nsamples]
    # print('pe', pe.min(), pe.max())
    mean_pe = np.mean(pe,axis=1)
    mean_pe2 = np.mean(pe*pe,axis=1)
    cv_per_npar = (mean_pe2 - mean_pe*mean_pe)/(T*T) /npar
    cv_per_npar_shifted = abs(cv_per_npar -cv_per_npar[0])
    print('pe shape',pe.shape, 'cv shape', cv_per_npar_shifted.shape)

    return cv_per_npar_shifted[-1]

if __name__ == '__main__':

    argv = sys.argv
    npar = int(argv[1])
    rho = argv[2]
    dim = int(argv[3])
    gamma = argv[4] #gas ##liguid liguid+gas
    saved_model = argv[5]
    dpt = int(argv[6])
    region = argv[7]
    n_split = int(argv[8])

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    data0_list = []
    data1_list = []
    data2_list = []
    temp_list = [0.44, 0.46, 0.48, 0.5]
    for temp in temp_list:

        load_file0 = f'../../../data_sets/gen_by_ML/lt0.1dpt100000_{region}/n{npar}rho{rho}T{temp}/energy_gamma{gamma}mb009_nsteps10000.pt'
        print('data0 list', load_file0)

        load_file1 = f'../../../data_sets/gen_by_MD/{dim}d/noML-metric-lt0.01every1t0.7t1000/n{npar}rho{rho}T{temp}/energy_gamma{gamma}_tmax1000.pt'
        print('data1 list', load_file1)

        load_file2 = f'../../../data_sets/gen_by_ML/lt0.1dpt{dpt}_{region}/n{npar}rho{rho}T{temp}/energy_gamma{gamma}mb{saved_model}_nsteps10000.pt'
        print('data2 list', load_file2)

        data1_list.append(load_file1)
        data2_list.append(load_file2)
        if (npar == 64 or npar == 128) and region == 'lg' :#and temp != 0.5:
            print('filename0 is ',temp)
            data0_list.append(load_file0)

    nsamples = 1000

    if (npar == 64 or npar == 128) and region == 'lg' :

        e0 = {}
        for i in range(len(data0_list)):
            print('load file', i, flush=True)
            data0 = torch.load(data0_list[i], map_location=map_location)
            e0["data0_" + str(i)] = data0["pe"].cpu().numpy()
            print(e0["data0_" + str(i)].shape)

        cv_mean0 = {}
        cv_mean0_append = []
        for i in range(len(data0_list)):
            for j in range(0, nsamples, n_split):
                print('samples', j, '....', j + n_split, 'T', i)
                cv_mean0["data0_" + str(i)] = Specific_heat(e0["data0_" + str(i)][:,j:j+n_split], npar, temp_list[i])  # shape [] ;take last time point
                cv_mean0_append.append(cv_mean0["data0_" + str(i)])

        cv_mu0 = []
        cv_se0 = []
        for k in range(len(temp_list)):
            cv_mu_mean0 = np.mean(
                cv_mean0_append[int(nsamples / n_split) * k : int(nsamples / n_split) + int(nsamples / n_split) * k])
            cv_mu_std0 = np.std(
                cv_mean0_append[int(nsamples / n_split) * k : int(nsamples / n_split) + int(nsamples / n_split) * k]) / np.sqrt(
                len(cv_mean0_append[int(nsamples / n_split) * k : int(nsamples / n_split) + int(nsamples / n_split) * k]))

            cv_mu0.append(cv_mu_mean0)
            cv_se0.append(cv_mu_std0)
            print(k, cv_mu_mean0, cv_mu_std0)

    e1 = {}
    for i in range(len(data1_list)):
        print('load file', i, flush=True)
        data1 = torch.load(data1_list[i], map_location=map_location)
        e1["data1_" + str(i)] = data1["pe"].cpu().numpy()
        print(e1["data1_" + str(i)].shape)

    e2 = {}
    for i in range(len(data2_list)):
        print('load file', i, flush=True)
        data2 = torch.load(data2_list[i], map_location=map_location)
        e2["data2_" + str(i)] = data2["pe"].cpu().numpy()
        print(e2["data2_" + str(i)].shape)

    cv_mean1 = {}
    cv_mean2 = {}
    cv_mean1_append = []
    cv_mean2_append = []

    for i in range(len(data1_list)):
        for j in range(0,nsamples,n_split):
            print('samples', j, '....', j + n_split)
            cv_mean1["data1_" + str(i)] = Specific_heat(e1["data1_" + str(i)][:,j:j+n_split], npar,temp_list[i]) # shape [] ;take last time point
            cv_mean2["data2_" + str(i)] = Specific_heat(e2["data2_" + str(i)][:,j:j+n_split], npar,temp_list[i]) # shape [] ;take last time point
            cv_mean1_append.append(cv_mean1["data1_" + str(i)])
            cv_mean2_append.append(cv_mean2["data2_" + str(i)])

    cv_mu1 = []
    cv_mu2 = []
    cv_se1 = []
    cv_se2 = []

    for k in range(len(temp_list)):

        cv_mu_mean1 = np.mean(cv_mean1_append[int(nsamples/n_split) * k : int(nsamples/n_split) + int(nsamples/n_split) * k])
        cv_mu_std1 = np.std(cv_mean1_append[int(nsamples/n_split) * k : int(nsamples/n_split) + int(nsamples/n_split) * k]) / np.sqrt(
                    len(cv_mean1_append[int(nsamples/n_split)*k:int(nsamples/n_split)+int(nsamples/n_split)*k]))

        # print('cv mean2',cv_mean2_append[int(1000/n_split)*k:int(1000/n_split)+int(1000/n_split)*k])
        cv_mu_mean2 = np.mean(cv_mean2_append[int(nsamples/n_split) * k : int(nsamples/n_split)+int(nsamples/n_split) * k])
        cv_mu_std2 = np.std(cv_mean2_append[int(nsamples/n_split) * k :
                    int(nsamples/n_split)+int(nsamples/n_split)*k]) / np.sqrt(
                    len(cv_mean2_append[int(nsamples/n_split)*k:int(nsamples/n_split)+int(nsamples/n_split)*k]))

        cv_mu1.append(cv_mu_mean1)
        cv_mu2.append(cv_mu_mean2)
        cv_se1.append(cv_mu_std1)
        cv_se2.append(cv_mu_std2)
        print(k, cv_mu_mean1, cv_mu_std1, cv_mu_mean2, cv_mu_std2)

    fig, axes = plt.subplots(1, 1, figsize=(6, 5))

    saved_txtfile = f'../../analysis/{dim}d/npar{npar}rho{rho}gamma{gamma}_cv.txt'

    # Check if the file exists, and remove it if it does
    if os.path.exists(saved_txtfile):
        os.remove(saved_txtfile)
        print(f"{saved_txtfile} was removed.")

    for j in np.arange(len(data1_list)):
        if (npar == 64 or npar == 128) and region == 'lg':#and j < len(data0_list):
            vv_mean = cv_mu1[j]
            vv_se = cv_se1[j]
            ml_mean = cv_mu2[j]
            ml_se = cv_se2[j]
            ml128_mean = cv_mu0[j]  # 64 or 128
            ml128_se = cv_se0[j]  # 64 or 128

            with open(saved_txtfile, 'a') as file:
                print('save file end of time point....')
                print('vv mean', vv_mean, 'se', vv_se,
                      'ml mean', ml_mean, 'se',ml_se ,
                      'ml128 mean', ml128_mean, 'se',ml128_se)
                file.write(f'{j} {vv_mean} {vv_se} {ml_mean} {ml_se} {ml128_mean} {ml128_se}\n')
        else:
            vv_mean = cv_mu1[j]
            vv_se = cv_se1[j]
            ml_mean = cv_mu2[j]
            ml_se = cv_se2[j]

            with open(saved_txtfile, 'a') as file:
                print('save file end of time point....')
                print('vv mean', vv_mean, 'se', vv_se,
                      'ml mean', ml_mean, 'se',ml_se)
                file.write(f'{j} {vv_mean} {vv_se} {ml_mean} {ml_se}\n')

    print(len(data0_list), len(data1_list), len(data2_list))
