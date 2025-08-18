import sys
sys.path.append('../../')

import torch
from utils.system_logs import system_logs
from utils.mydevice import mydevice
import math
import numpy as np
#import MD.pairwise_distance as plot_pairs
from utils.pbc import pairwise_dq_pbc
from utils.get_paired_distance_indices import get_paired_distance_indices
from collections import Counter


def pack_data(qpl_list):
    # shape = [nsamples, (q,p,boxsize), trajetory,  nparticles, DIM]
    q_traj = qpl_list[:, 0, :, :, :].clone().detach()
    p_traj = qpl_list[:, 1, :, :, :].clone().detach()
    l_init = qpl_list[:, 2, :, :, :].clone().detach()

    return q_traj, p_traj, l_init


def l_max_distance(l_list):
    boxsize = torch.mean(l_list)
    L_h = boxsize / 2.
    q_max = math.sqrt(L_h * L_h + L_h * L_h + L_h * L_h)
    print('boxsize', boxsize.item(), 'maximum distance dq = {:.2f}, dq^2 = {:.2f}'.format(q_max, q_max * q_max))
    return boxsize, q_max


def pairDistributionFunction(grdelta, grbinmax, q_list, l_list):
    """Generates a pair-distribution function on given data"""

    # q_list shape [nsample, nparticle, dim]
    nsamples = q_list.shape[0]
    nparticles = q_list.shape[1]
    dim = q_list.shape[2]
    boxsize = torch.mean(l_list)
    halfboxsize = boxsize / 2.
    # print("Generating rdf..."),

    dq = pairwise_dq_pbc(q_list, l_list)  # shape = [nsamples,nparticles,nparticles,dim]
    idx = get_paired_distance_indices.get_indices(dq.shape)  # offset i particle, i particle
    dr = get_paired_distance_indices.reduce(dq, idx)
    dr = dr.view([nsamples, nparticles, nparticles - 1, dim])

    r = torch.sqrt(torch.sum(dr * dr, dim=-1))
    # r shape [nsample, nparticle, nparticle-1]
    # print('dq min {:.2f}; max {:.2f}'.format(torch.min(dr).item(),torch.max(dr).item()))
    # plot_pairs.plot_pairs(r,npar,rho, boxsize)

    r = r.view(nsamples, -1)
    # r shape [nsample, nparticle*nparticle-1]
    # print('grdelta',  grdelta, 'half boxsize', halfboxsize.item())
    assert ((r < math.sqrt(halfboxsize * halfboxsize + halfboxsize * halfboxsize + halfboxsize * halfboxsize)).all() == True), 'rij go out box'

    # indices = torch.where(r < halfboxsize)
    grbin = (r / grdelta).to(torch.long)  # bin corresponding to the interval (r,r+dr)
    indices = torch.where(grbin < grbinmax)

    count = Counter(grbin[indices].tolist())
    sorted_key = list(dict(sorted(count.items())))
    sorted_value = list(dict(sorted(count.items())).values())
    mean_sorted_value = [hist / nsamples for hist in sorted_value]

    return sorted_key, mean_sorted_value


if __name__ == '__main__':
    # python rdf.py 64 0.85 0.9 3 1000 0.002 0 0 None 1800000 l

    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    argv = sys.argv
    if len(argv) != 12:
        print('usage <programe> <npar> <rho> <temp> <dim> <tau_long> <nstep> <gamma> <saved_model> <dpt> <region>')
        quit()

    npar = argv[1]
    rho = argv[2]
    T = argv[3]
    dim = int(argv[4])
    n_split = int(argv[5])
    tau_long = float(argv[6])
    saved_tp = int(argv[7])
    gamma = int(argv[8])
    saved_model = argv[9]
    dpt = int(argv[10])
    region = argv[11]

    if saved_model.strip() == "None":
       saved_model = None
    else:
       saved_model = saved_model.strip()

    if saved_model is None:
        print('run md data .......')
        data = {
            "filename": '../../../data_sets/gen_by_MD/{}d/noML-metric-st1e-4every0.1t8/n{}rho{}T{}/'.format(dim,  npar, rho, T)
                + 'n{}rho{}T{}.pt'.format(npar, rho, T),
            "saved_dir": '../../../data_sets/gen_by_MD/{}d/noML-metric-st1e-4every0.1t8/n{}rho{}T{}/'.format(dim,  npar, rho,  T)
            # "filename": '../../../data_sets/gen_by_MD/{}d/n{}lt0.1stpstraj18_l_dpt45000.pt'.format(dim,npar),
            # "saved_dir": '../../../data_sets/gen_by_MD/{}d/'.format(dim)
            #              + 'n{}rho{}T{}gamma{}.pt'.format(npar, rho, T, gamma),
            # "filename": '../../../data_sets/gen_by_MD/{}d/noML-metric-lt{}every1t1000/n{}rho{}T{}/'.format(dim,tau_long, npar, rho, T)
            #            + 'n{}rho{}T{}gamma{}.pt'.format(npar, rho, T, gamma),
            # "saved_dir": '../../../data_sets/gen_by_MD/{}d/noML-metric-lt{}every1t1000/n{}rho{}T{}/'.format(dim, tau_long, npar, rho, T)
            # "filename": '../../../data_sets/gen_by_MC/{}d/n{}rho{}/'.format(dim, npar, rho )
            #             + 'n{}T{}seed1386nsamples1.pt'.format(npar, T),
            # "saved_dir": '../../../data_sets/gen_by_MC/{}d/n{}rho{}/'.format(dim,  npar, rho )
        }
    else:
        print('run ml data  .......')
        data = {
            "filename": '../../../data_sets/gen_by_ML/{}d/lt{}dpt{}_{}/n{}rho{}T{}/'.format(dim,tau_long,dpt, region, npar, rho, T)
                        + 'pred_n{}len08ws08gamma{}LUF{}_tau{}.pt'.format(npar, gamma, saved_model,tau_long),
            "saved_dir": '../../../data_sets/gen_by_ML/3d/lt{}dpt{}_{}/n{}rho{}T{}/'.format(tau_long,dpt, region, npar, rho, T)}

    maindict = {
        "grdelta": 0.06,
        "traj_len": 8
    }

    print('load file', data, flush=True)
    data1 = torch.load(data["filename"], map_location=map_location)
    print('shape', data1['qpl_trajectory'].shape)
    print('saved dir....', data["saved_dir"])

    traj_len = maindict["traj_len"]
    input_seq_idx = traj_len - 1
    grdelta = maindict["grdelta"]

    dd = {}
    nsamples = data1['qpl_trajectory'].shape[0]

    for j in range(0, nsamples, n_split):
        print('samples', j, '....', j + n_split)
        tmp = data1['qpl_trajectory'][j:j + n_split, :, 0:saved_tp + 1, :, :]
        # shape = [nsamples, (q,p,boxsize), trajetory,  nparticles, DIM]
        print('qpl traj shape', tmp.shape, 'saved_tp ', saved_tp, flush=True)

        dd["q_traj_s" + str(j)], _, dd["l_s" + str(j)] = pack_data(tmp)
        print('shape', dd["q_traj_s{}".format(j)].shape, dd["l_s" + str(j)].shape, torch.mean(dd["l_s" + str(j)]),
              flush=True)
        # shape = [nsamples,  trajectory,  nparticles, DIM]
        boxsize, q_max = l_max_distance(dd["l_s" + str(j)])

    _, traj_len, nparticles, _ = dd["q_traj_s0"].shape

    # tmp_file = []
    # l_list = []
    # for j in range(0,nsamples,n_split):
    #     print('samples', j, '....', j+n_split)
    #     tmp_file.append(dd["q_traj_s" + str(j)])
    #     l_list.append(dd["l_s" + str(j)])
    #
    # q_traj  = torch.cat(tmp_file,dim=0)
    # l_list  = torch.cat(l_list,dim=0)
    # print( 'q traj shape ', q_traj.shape, 'l list shape', l_list.shape)

    # grbinmax = int(q_max / grdelta)

    grbinmax = int((boxsize / 2) / grdelta)
    print('boxsize', boxsize.item(), 'grdelta', grdelta, 'grbinmax', grbinmax, 'saved_tp', saved_tp)

    for j in range(0, nsamples, n_split):
        print('samples', j, '....', j + n_split)
        time = list(range(0, saved_tp + 1))

        r_ij = []
        nrmid_step = []
        ngr_step = []
        count = 0

        for step in time:

            sorted_grbin, mean_sorted_value = pairDistributionFunction(grdelta, grbinmax,
                                                                       dd["q_traj_s" + str(j)][:, step],
                                                                       dd["l_s" + str(j)][:, step])

            nrmid = []
            ngr = []

            # normalize gr
            for grbin, value in zip(sorted_grbin, mean_sorted_value):
                rinner = grbin * grdelta
                router = rinner + grdelta

                if dim == 2:
                    shellvol = 2 * math.pi * rinner * grdelta  # math.pi * (router**2 - rinner**2)
                    gr = (boxsize ** 2 / nparticles) / (nparticles - 1) * value / shellvol

                elif dim == 3:
                    shellvol = 4 * math.pi * (rinner ** 2) * grdelta  # or (4/3)*pi*(router**3 - rinner**3)
                    gr = (boxsize ** 3 / nparticles) / (nparticles - 1) * value / shellvol

                else:
                    raise ValueError(f"Dimension {dim} not supported....")

                rmid = rinner + 0.5 * grdelta
                r_ij.append(rmid)
                nrmid.append(rmid)
                ngr.append(gr.item())

            # print('count ', count, ' len gr', len(ngr),'sum gr', sum(ngr))
            count += 1
            nrmid_step.append(nrmid)
            ngr_step.append(ngr)

        print('min r', min(r_ij), 'max r', max(r_ij))

        if nsamples == n_split:
            if saved_model is None:
                print('save dir....', data["saved_dir"]+'rij_gr_gamma{}.pt'.format(gamma))
                torch.save({'nrmid': nrmid_step, 'ngr' : ngr_step}, data["saved_dir"] +'rij_gr_gamma{}.pt'.format(gamma))
            else:
                print('save dir....', data["saved_dir"]+'rij_gr_gamma{}LUF{}.pt'.format(gamma,saved_model))
                torch.save({'nrmid': nrmid_step, 'ngr' : ngr_step}, data["saved_dir"] +'rij_gr_gamma{}LUF{}.pt'.format(gamma,saved_model))

        else:
            if saved_model is None:
                print('save dir....', data["saved_dir"] + 'rij_gr_gamma{}_s{}.pt'.format(gamma, int((j + n_split) / n_split)))
                torch.save({'nrmid': nrmid_step, 'ngr': ngr_step}, data["saved_dir"] + 'rij_gr_gamma{}_s{}.pt'.format(gamma, int((j + n_split) / n_split)))
            else:
                print('save dir....', data["saved_dir"] + 'rij_gr_gamma{}LUF{}_s{}.pt'.format(gamma, saved_model, int((j + n_split) / n_split)))
                torch.save({'nrmid': nrmid_step, 'ngr': ngr_step}, data["saved_dir"] + 'rij_gr_gamma{}LUF{}_s{}.pt'.format(gamma, saved_model,
                                                                                      int((j + n_split) / n_split)))

