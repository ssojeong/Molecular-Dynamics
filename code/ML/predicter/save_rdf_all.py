import sys 
sys.path.append( '../../')

import torch
from utils.system_logs              import system_logs
from utils.mydevice                 import mydevice
import math
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    # python save_rdf_all.py 128 0.3 0.1 10000 10 485 800000 lg 200

    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    argv = sys.argv

    npar = int(argv[1])
    rho = argv[2]
    dim = int(argv[3])
    gamma = argv[4]
    saved_model = argv[5]
    dpt = int(argv[6])
    region = argv[7]
    n_split = int(argv[8])

    data0_list = []
    data1_list = []
    data2_list = []
    temp_list = [0.44, 0.46, 0.48, 0.5]

    for temp in temp_list:

        data = {
                "filename0": '../../../data_sets/gen_by_ML/lt0.1dpt100000_{}/n{}rho{}T{}/'.format(region, npar, rho, temp)
                         + f'rij_gr_gamma{gamma}mb009',
                "filename1" : '../../../data_sets/gen_by_MD/{}d/noML-metric-lt0.01every1t0.7t1000/n{}rho{}T{}/'.format(dim, npar,rho,temp)
                 + f'rij_gr_gamma{gamma}',
                "filename2": '../../../data_sets/gen_by_ML/lt0.1dpt{}_{}/n{}rho{}T{}/'.format(dpt, region, npar, rho, temp)
                + f'rij_gr_gamma{gamma}mb{saved_model}'
                }

        data1_list.append(data['filename1'])
        data2_list.append(data['filename2'])

        if (npar == 64 or npar == 128) and region == 'lg' : #and temp != 0.5:
            print('filename0 is ',temp)
            data0_list.append(data['filename0'])

    nsamples = 1000 # test set

    d1 = {}
    for i in range(len(data1_list)):
        print('load file', i, flush=True)
        for j in range(0, nsamples, n_split):
            print('samples', j, '....', j + n_split, 'T=', j)
            d1["data1_" + str(i) + "_" + str(j)] = torch.load(data1_list[i] + f'_s{int((n_split+j)/n_split)}.pt',map_location=map_location)

    dd1 = {}

    rmid_1st_mu0_all = []
    rmid_1st_se0_all = []
    rmid_2nd_mu0_all = []
    rmid_2nd_se0_all = []

    gr_1st_mu0_all = []
    gr_1st_se0_all = []
    gr_2nd_mu0_all = []
    gr_2nd_se0_all = []

    rmid_1st_mu1_all = []
    rmid_1st_se1_all = []
    rmid_2nd_mu1_all = []
    rmid_2nd_se1_all = []

    gr_1st_mu1_all = []
    gr_1st_se1_all = []
    gr_2nd_mu1_all = []
    gr_2nd_se1_all = []

    for i in range(len(data1_list)):

        rmid_first_mean0_append = []
        rmid_first_mean1_append = []
        gr_first_mean0_append = []
        gr_first_mean1_append = []

        rmid_second_mean0_append = []
        rmid_second_mean1_append = []
        gr_second_mean0_append = []
        gr_second_mean1_append = []

        for j in range(0, nsamples, n_split):

            dd1["rmid" + str(i)+ "_" + str(j)] = d1["data1_" + str(i)+ "_" + str(j)]['nrmid'] # save nrmid over time steps
            dd1["gr" + str(i)+ "_" + str(j)] = d1["data1_" + str(i)+ "_" + str(j)]['ngr'] # save ngr over time steps

            ngr0 = np.array(dd1["rmid" + str(i)+ "_" + str(j)][0]) # take rmid at initial time
            mc = np.array(dd1["gr" + str(i)+ "_" + str(j)][0]) # take gr at initial time
            ngr1 = np.array(dd1["rmid" + str(i) + "_" + str(j)][-1]) # take rmid at last time
            vv = np.array(dd1["gr" + str(i) + "_" + str(j)][-1]) # take gr at last time

            print(ngr0.shape, mc.shape, ngr1.shape, vv.shape)

            max_0_idx = np.where(mc == np.max(mc))
            max_1_idx = np.where(vv == np.max(vv))

            print('first rmid', ngr0[max_0_idx],ngr1[max_1_idx])
            print('first gr', np.max(mc), np.max(vv))

            rmid_first_mean0_append.append(ngr0[max_0_idx])
            rmid_first_mean1_append.append(ngr1[max_1_idx])
            gr_first_mean0_append.append(mc[max_0_idx])
            gr_first_mean1_append.append(vv[max_1_idx])

            second0_idx = np.where((ngr0 > 2) & (ngr0 < 2.25))[0]
            second1_idx = np.where((ngr1 > 2) & (ngr1 < 2.25))[0]

            max_second0_idx = np.where(mc == mc[second0_idx].max())[0][0]
            max_second1_idx = np.where(vv == vv[second1_idx].max())[0][0]

            print('second rmid', ngr0[max_second0_idx], ngr1[max_second1_idx])
            print('second gr', mc[max_second0_idx], vv[max_second1_idx])

            rmid_second_mean0_append.append(ngr0[max_second0_idx])
            rmid_second_mean1_append.append(ngr1[max_second1_idx])
            gr_second_mean0_append.append(mc[max_second0_idx])
            gr_second_mean1_append.append(vv[max_second1_idx])

        print('==== first peak over 5 groups .....=====')
        print(gr_first_mean0_append, flush=True)
        print(gr_first_mean1_append, flush=True)

        rmid_first_mu0 = np.mean(rmid_first_mean0_append)
        rmid_first_se0 = np.std(rmid_first_mean0_append) / np.sqrt(len(rmid_first_mean0_append))
        print('mc mean/std first rmid',rmid_first_mu0, rmid_first_se0)

        gr_first_mu0 = np.mean(gr_first_mean0_append)
        gr_first_se0 = np.std(gr_first_mean0_append) / np.sqrt(len(gr_first_mean0_append))
        print('mc mean/std first gr', gr_first_mu0, gr_first_se0)

        rmid_first_mu1 = np.mean(rmid_first_mean1_append)
        rmid_first_se1 = np.std(rmid_first_mean1_append) / np.sqrt(len(rmid_first_mean1_append))
        print('vv mean/std first rmid',rmid_first_mu1, rmid_first_se1)

        gr_first_mu1 = np.mean(gr_first_mean1_append)
        gr_first_se1 = np.std(gr_first_mean1_append) / np.sqrt(len(gr_first_mean1_append))
        print('vv mean/std first gr', gr_first_mu1, gr_first_se1)

        print('==== second peak over 5 groups .....=====')
        print(gr_second_mean0_append, flush=True)
        print(gr_second_mean1_append, flush=True)

        rmid_second_mu0 = np.mean(rmid_second_mean0_append)
        rmid_second_se0 = np.std(rmid_second_mean0_append) / np.sqrt(len(rmid_second_mean0_append))
        print('mc mean/std second rmid', rmid_second_mu0, rmid_second_se0)

        gr_second_mu0 = np.mean(gr_second_mean0_append)
        gr_second_se0 = np.std(gr_second_mean0_append) / np.sqrt(len(gr_second_mean0_append))
        print('mc mean/std second gr', gr_second_mu0, gr_second_se0)

        rmid_second_mu1 = np.mean(rmid_second_mean1_append)
        rmid_second_se1 = np.std(rmid_second_mean1_append) / np.sqrt(len(rmid_second_mean1_append))
        print('vv mean/std second rmid',rmid_second_mu1, rmid_second_se1)

        gr_second_mu1 = np.mean(gr_second_mean1_append)
        gr_second_se1 = np.std(gr_second_mean1_append) / np.sqrt(len(gr_second_mean1_append))
        print('vv mean/std second gr',gr_second_mu1, gr_second_se1)

        rmid_1st_mu0_all.append(rmid_first_mu0)
        rmid_1st_se0_all.append(rmid_first_se0)
        gr_1st_mu0_all.append(gr_first_mu0)
        gr_1st_se0_all.append(gr_first_se0)

        rmid_2nd_mu0_all.append(rmid_second_mu0)
        rmid_2nd_se0_all.append(rmid_second_se0)
        gr_2nd_mu0_all.append(gr_second_mu0)
        gr_2nd_se0_all.append(gr_second_se0)

        rmid_1st_mu1_all.append(rmid_first_mu1)
        rmid_1st_se1_all.append(rmid_first_se1)
        gr_1st_mu1_all.append(gr_first_mu1)
        gr_1st_se1_all.append(gr_first_se1)

        rmid_2nd_mu1_all.append(rmid_second_mu1)
        rmid_2nd_se1_all.append(rmid_second_se1)
        gr_2nd_mu1_all.append(gr_second_mu1)
        gr_2nd_se1_all.append(gr_second_se1)

    d2 = {}
    for i in range(len(data2_list)):
        print('load file', i, flush=True)  # 0 for T=0.46, 1 for T=0.48, 2 for T=0.55
        for j in range(0, nsamples, n_split):
            print('samples', j, '....', j + n_split, 'T=', j)
            d2["data2_" + str(i) + "_" + str(j)] = torch.load(data2_list[i] + f'_s{int((n_split+j)/n_split)}.pt',map_location=map_location)

    dd2 = {}

    rmid_1st_mu2_all = []
    rmid_1st_se2_all = []
    rmid_2nd_mu2_all = []
    rmid_2nd_se2_all = []

    gr_1st_mu2_all = []
    gr_1st_se2_all = []
    gr_2nd_mu2_all = []
    gr_2nd_se2_all = []

    for i in range(len(data2_list)):

        rmid_first_mean2_append = []
        gr_first_mean2_append = []

        rmid_second_mean2_append = []
        gr_second_mean2_append = []

        for j in range(0, nsamples, n_split):

            dd2["rmid" + str(i)+ "_" + str(j)] = d2["data2_" + str(i)+ "_" + str(j)]['nrmid']
            dd2["gr" + str(i)+ "_" + str(j)] = d2["data2_" + str(i)+ "_" + str(j)]['ngr']
            ngr3 = np.array(dd2["rmid" + str(i) + "_" + str(j)][-1])
            ml = np.array(dd2["gr" + str(i) + "_" + str(j)][-1])

            print(ngr3.shape,ml.shape)

            max_3_idx = np.where(ml == np.max(ml))

            print('first rmid', ngr3[max_3_idx])
            print('first gr', np.max(ml))
            rmid_first_mean2_append.append(ngr3[max_3_idx])
            gr_first_mean2_append.append(ml[max_3_idx])

            second3_idx = np.where((ngr3 > 2) & (ngr3 < 2.25))[0]
            # print(second0_idx, mc[second0_idx], mc[second0_idx].max())
            # print(np.where(mc == mc[second0_idx].max())[0][0])

            max_second3_idx = np.where(ml == ml[second3_idx].max())[0][0]

            print('second rmid', ngr3[max_second3_idx])
            print('second gr', ml[max_second3_idx])

            rmid_second_mean2_append.append(ngr3[max_second3_idx])
            gr_second_mean2_append.append(ml[max_second3_idx])


        print('==== first peak over 5 groups .....=====')
        print(gr_first_mean2_append, flush=True)

        rmid_first_mu2 = np.mean(rmid_first_mean2_append)
        rmid_first_se2 = np.std(rmid_first_mean2_append) / np.sqrt(len(rmid_first_mean2_append))
        print('ml mean/std first rmid',rmid_first_mu2, rmid_first_se2)

        gr_first_mu2 = np.mean(gr_first_mean2_append)
        gr_first_se2 = np.std(gr_first_mean2_append) / np.sqrt(len(gr_first_mean2_append))
        print('ml mean/std first gr', gr_first_mu2, gr_first_se2)

        print('==== second peak over 5 groups .....=====')
        print(gr_second_mean2_append, flush=True)

        rmid_second_mu2 = np.mean(rmid_second_mean2_append)
        rmid_second_se2 = np.std(rmid_second_mean2_append) / np.sqrt(len(rmid_second_mean2_append))
        print('ml mean/std second rmid',rmid_second_mu2, rmid_second_se2)

        gr_second_mu2 = np.mean(gr_second_mean2_append)
        gr_second_se2 = np.std(gr_second_mean2_append) / np.sqrt(len(gr_second_mean2_append))
        print('ml mean/std second gr', gr_second_mu2, gr_second_se2)

        rmid_1st_mu2_all.append(rmid_first_mu2)
        rmid_1st_se2_all.append(rmid_first_se2)
        gr_1st_mu2_all.append(gr_first_mu2)
        gr_1st_se2_all.append(gr_first_se2)

        rmid_2nd_mu2_all.append(rmid_second_mu2)
        rmid_2nd_se2_all.append(rmid_second_se2)
        gr_2nd_mu2_all.append(gr_second_mu2)
        gr_2nd_se2_all.append(gr_second_se2)

    print('')
    if (npar == 64 or npar == 128) and region == 'lg'  :

        d3 = {}
        for i in range(len(data0_list)):
            print('load file', i, flush=True)  # 0 for T=0.46, 1 for T=0.48, 2 for T=0.55
            for j in range(0, nsamples, n_split):
                print('samples', j, '....', j + n_split, 'T=', j)
                d3["data3_" + str(i) + "_" + str(j)] = torch.load(
                    data0_list[i] + f'_s{int((n_split + j) / n_split)}.pt', map_location=map_location)

        dd3 = {}

        rmid_1st_mu3_all = []
        rmid_1st_se3_all = []
        rmid_2nd_mu3_all = []
        rmid_2nd_se3_all = []

        gr_1st_mu3_all = []
        gr_1st_se3_all = []
        gr_2nd_mu3_all = []
        gr_2nd_se3_all = []

        for i in range(len(data0_list)):

            rmid_first_mean3_append = []
            gr_first_mean3_append = []

            rmid_second_mean3_append = []
            gr_second_mean3_append = []

            for j in range(0, nsamples, n_split):
                dd3["rmid" + str(i) + "_" + str(j)] = d3["data3_" + str(i) + "_" + str(j)]['nrmid']
                dd3["gr" + str(i) + "_" + str(j)] = d3["data3_" + str(i) + "_" + str(j)]['ngr']
                ngr128 = np.array(dd3["rmid" + str(i) + "_" + str(j)][-1])
                ml128 = np.array(dd3["gr" + str(i) + "_" + str(j)][-1])

                print(ngr128.shape, ml128.shape)

                max_4_idx = np.where(ml128 == np.max(ml128))

                print('first rmid', ngr128[max_4_idx])
                print('first gr', np.max(ml128))
                rmid_first_mean3_append.append(ngr128[max_4_idx])
                gr_first_mean3_append.append(ml128[max_4_idx])

                second4_idx = np.where((ngr128 > 2) & (ngr128 < 2.25))[0]
                # print(second0_idx, mc[second0_idx], mc[second0_idx].max())
                # print(np.where(mc == mc[second0_idx].max())[0][0])

                max_second4_idx = np.where(ml128 == ml128[second4_idx].max())[0][0]

                print('second rmid', ngr128[max_second4_idx])
                print('second gr', ml128[max_second4_idx])

                rmid_second_mean3_append.append(ngr128[max_second4_idx])
                gr_second_mean3_append.append(ml128[max_second4_idx])

            print('==== first peak over 5 groups .....=====')
            print(gr_first_mean3_append, flush=True)

            rmid_first_mu3 = np.mean(rmid_first_mean3_append)
            rmid_first_se3 = np.std(rmid_first_mean3_append) / np.sqrt(len(rmid_first_mean3_append))
            print('ml128 mean/std first rmid', rmid_first_mu3, rmid_first_se3)

            gr_first_mu3 = np.mean(gr_first_mean3_append)
            gr_first_se3 = np.std(gr_first_mean3_append) / np.sqrt(len(gr_first_mean3_append))
            print('ml128 mean/std first gr', gr_first_mu3, gr_first_se3)

            print('==== second peak over 5 groups .....=====')
            print(gr_second_mean3_append, flush=True)

            rmid_second_mu3 = np.mean(rmid_second_mean3_append)
            rmid_second_se3 = np.std(rmid_second_mean3_append) / np.sqrt(len(rmid_second_mean3_append))
            print('ml128 mean/std second rmid', rmid_second_mu3, rmid_second_se3)

            gr_second_mu3 = np.mean(gr_second_mean3_append)
            gr_second_se3 = np.std(gr_second_mean3_append) / np.sqrt(len(gr_second_mean3_append))
            print('ml128 mean/std second gr',gr_second_mu3, gr_second_se3)

            rmid_1st_mu3_all.append(rmid_first_mu3)
            rmid_1st_se3_all.append(rmid_first_se3)
            gr_1st_mu3_all.append(gr_first_mu3)
            gr_1st_se3_all.append(gr_first_se3)

            rmid_2nd_mu3_all.append(rmid_second_mu3)
            rmid_2nd_se3_all.append(rmid_second_se3)
            gr_2nd_mu3_all.append(gr_second_mu3)
            gr_2nd_se3_all.append(gr_second_se3)

    saved_txtfile = f'../../analysis/{dim}d/npar{npar}rho{rho}gamma{gamma}_rdf.txt'

    # Check if the file exists, and remove it if it does
    if os.path.exists(saved_txtfile):
        os.remove(saved_txtfile)
        print(f"{saved_txtfile} was removed.")

    for j in np.arange(len(temp_list)):
        # ax.set_title(title[j])

        mc_1st_rmid_mu = rmid_1st_mu0_all[j]
        mc_1st_rmid_se = rmid_1st_se0_all[j]
        mc_1st_gr_mu = gr_1st_mu0_all[j]
        mc_1st_gr_se = gr_1st_se0_all[j]

        mc_2nd_rmid_mu = rmid_2nd_mu0_all[j]
        mc_2nd_rmid_se = rmid_2nd_se0_all[j]
        mc_2nd_gr_mu = gr_2nd_mu0_all[j]
        mc_2nd_gr_se = gr_2nd_se0_all[j]

        vv_1st_rmid_mu = rmid_1st_mu1_all[j]
        vv_1st_rmid_se = rmid_1st_se1_all[j]
        vv_1st_gr_mu = gr_1st_mu1_all[j]
        vv_1st_gr_se = gr_1st_se1_all[j]

        vv_2nd_rmid_mu = rmid_2nd_mu1_all[j]
        vv_2nd_rmid_se = rmid_2nd_se1_all[j]
        vv_2nd_gr_mu = gr_2nd_mu1_all[j]
        vv_2nd_gr_se = gr_2nd_se1_all[j]

        ml_1st_rmid_mu = rmid_1st_mu2_all[j]
        ml_1st_rmid_se = rmid_1st_se2_all[j]
        ml_1st_gr_mu = gr_1st_mu2_all[j]
        ml_1st_gr_se = gr_1st_se2_all[j]

        ml_2nd_rmid_mu = rmid_2nd_mu2_all[j]
        ml_2nd_rmid_se = rmid_2nd_se2_all[j]
        ml_2nd_gr_mu = gr_2nd_mu2_all[j]
        ml_2nd_gr_se = gr_2nd_se2_all[j]

        print('mc 1st', mc_1st_rmid_mu, mc_1st_rmid_se, mc_1st_gr_mu, mc_1st_gr_se)
        print('mc 2nd', mc_2nd_rmid_mu, mc_2nd_rmid_se, mc_2nd_gr_mu, mc_2nd_gr_se)
        print('vv 1st', vv_1st_rmid_mu, vv_1st_rmid_se, vv_1st_gr_mu, vv_1st_gr_se)
        print('vv 2nd', vv_2nd_rmid_mu, vv_2nd_rmid_se, vv_2nd_gr_mu, vv_2nd_gr_se)
        print('ml 1st', ml_1st_rmid_mu, ml_1st_rmid_se, ml_1st_gr_mu, ml_1st_gr_se)
        print('ml 2nd', ml_2nd_rmid_mu, ml_2nd_rmid_se, ml_2nd_gr_mu, ml_2nd_gr_se)


        if (npar == 64 or npar == 128) and region == 'lg'  :#and j < len(data1_list)-1 :

            ml128_1st_rmid_mu = rmid_1st_mu3_all[j]
            ml128_1st_rmid_se = rmid_1st_se3_all[j]
            ml128_1st_gr_mu = gr_1st_mu3_all[j]
            ml128_1st_gr_se = gr_1st_se3_all[j]

            ml128_2nd_rmid_mu = rmid_2nd_mu3_all[j]
            ml128_2nd_rmid_se = rmid_2nd_se3_all[j]
            ml128_2nd_gr_mu = gr_2nd_mu3_all[j]
            ml128_2nd_gr_se = gr_2nd_se3_all[j]


            with open(saved_txtfile, 'a') as file:
                file.write(f'{j} {mc_1st_rmid_mu} {mc_1st_rmid_se} {mc_1st_gr_mu} {mc_1st_gr_se} '
                           f'{mc_2nd_rmid_mu} {mc_2nd_rmid_se} {mc_2nd_gr_mu} {mc_2nd_gr_se} '
                           f'{vv_1st_rmid_mu} {vv_1st_rmid_se} {vv_1st_gr_mu} {vv_1st_gr_se} '
                           f'{vv_2nd_rmid_mu} {vv_2nd_rmid_se} {vv_2nd_gr_mu} {vv_2nd_gr_se} '
                           f'{ml_1st_rmid_mu} {ml_1st_rmid_se} {ml_1st_gr_mu} {ml_1st_gr_se} '
                           f'{ml_2nd_rmid_mu} {ml_2nd_rmid_se} {ml_2nd_gr_mu} {ml_2nd_gr_se} '
                           f'{ml128_1st_rmid_mu} {ml128_1st_rmid_se} {ml128_1st_gr_mu} {ml128_1st_gr_se} '
                           f'{ml128_2nd_rmid_mu} {ml128_2nd_rmid_se} {ml128_2nd_gr_mu} {ml128_2nd_gr_se} \n')

        else:
            with open(saved_txtfile, 'a') as file:
                file.write(f'{j} {mc_1st_rmid_mu} {mc_1st_rmid_se} {mc_1st_gr_mu} {mc_1st_gr_se} '
                           f'{mc_2nd_rmid_mu} {mc_2nd_rmid_se} {mc_2nd_gr_mu} {mc_2nd_gr_se} '
                           f'{vv_1st_rmid_mu} {vv_1st_rmid_se} {vv_1st_gr_mu} {vv_1st_gr_se} '
                           f'{vv_2nd_rmid_mu} {vv_2nd_rmid_se} {vv_2nd_gr_mu} {vv_2nd_gr_se} '
                           f'{ml_1st_rmid_mu} {ml_1st_rmid_se} {ml_1st_gr_mu} {ml_1st_gr_se} '
                           f'{ml_2nd_rmid_mu} {ml_2nd_rmid_se} {ml_2nd_gr_mu} {ml_2nd_gr_se} \n')


    print(len(data0_list),len(data1_list),len(data2_list),len(data2_list))