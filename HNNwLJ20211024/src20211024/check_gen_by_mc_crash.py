from phase_space.phase_space         import phase_space
from hamiltonian.noML_hamiltonian    import noML_hamiltonian
from utils.data_io                   import data_io
import matplotlib.pyplot as plt

import os
import sys
import json
import torch
import math

def save_crash_config(crsh_q, crsh_p, boxsize, crash_path, info):

    if not os.path.exists(crash_path):
        os.makedirs(crash_path)

    crash_filename = crash_path + 'soft-crashed_' + str(info) + '.pt'

    # write q, p list
    qp_list = torch.stack((crsh_q, crsh_p), dim=1)
    qp_list = torch.unsqueeze(qp_list, dim=2)
    data_io.write_trajectory_qp(crash_filename, qp_list, boxsize)

def thrsh(rthrsh, pthrsh):

    pothrsh = 4 * (1 / (rthrsh) ** 12 - 1 / (rthrsh) ** 6)
    dhdqthrsh = 4 * ((-12) / (rthrsh) ** 13) - 4 * ((-6) / (rthrsh) ** 7)

    kethrsh = pthrsh * pthrsh / 2 + pthrsh * pthrsh / 2
    ethrsh = kethrsh + pothrsh
    print('rthrsh', rthrsh, 'pothrsh', pothrsh, 'dhdqthrsh', dhdqthrsh, 'pthrsh', pthrsh, 'kethrsh', kethrsh)
    return rthrsh, pothrsh, dhdqthrsh, pthrsh, kethrsh, ethrsh

if __name__ == '__main__':

    '''python check_gen_by_mc2.py jsonfile1.dict no'''

    argv = sys.argv
    json_file = argv[1]
    is_crash = argv[2]

    with open(json_file) as f:
        data = json.load(f)

    phase_space = phase_space()
    hamiltonian_obj = noML_hamiltonian()

    filelist = data['filelist']
    rho = data['label']
    crash_path = data['crash_path']

    init_e1 = {}
    ke1 = {}
    p_dist1 = {}
    init_u1 = {}
    f1_magnitude1 = {}
    # rho = ['0.38T0.27', '0.38T0.47', '0.38T0.71']
    # rho = [0.10, 0.14, 0.2, 0.27, 0.38]

    for i in range(len(filelist)):

        all_idx = []
        crash_flag = False

        infile = torch.load(filelist[i])

        tau_short = infile['tau_short']
        tau_long = infile['tau_long']
        boxsize  = infile['boxsize']
        qp_traj = infile['qp_trajectory']

        init_qp = qp_traj[:, :, 0, :, :]
        # init_qp.shape = [nsamples, (q, p), nparticle, DIM]

        p_list = init_qp[:, 1, :, :]

        _, _, npar, DIM = init_qp.shape
        terms = hamiltonian_obj.get_terms()

        # initial state
        phase_space.set_q(init_qp[:, 0, :, :])
        phase_space.set_p(init_qp[:, 1, :, :])
        phase_space.set_boxsize(boxsize)

        init_u = terms[0].energy(phase_space) / npar
        init_e = hamiltonian_obj.total_energy(phase_space) / npar
        # init_e.shape = [nsamples]

        ########################### thrsh  ##################################
        if is_crash == 'no' :
            q_list = init_qp[:, 0, :, :] / boxsize
            _, d = phase_space.paired_distance_reduced(q_list, npar, DIM)
            d = d * boxsize
            rthrsh = torch.min(d)
            pthrsh = math.sqrt(2 * 1.0) * math.sqrt(-1. * math.log(math.sqrt(2 * math.pi) * 1e-5))
            rthrsh, pothrsh, dhdqthrsh, pthrsh, kethrsh, ethrsh = thrsh(rthrsh, pthrsh)

        else:
            #rthrsh = 0.9146
            rthrsh =  0.8632674463786768
            pthrsh =  4.876708623335836
            # q_list = init_qp[:, 0, :, :] / boxsize
            # _, d = phase_space.paired_distance_reduced(q_list, npar, DIM)
            # d = d * boxsize
            # rthrsh = torch.min(d)
            rthrsh, pothrsh, dhdqthrsh, pthrsh, kethrsh, ethrsh = thrsh(rthrsh, pthrsh)

        ###################### check min-max f1 ############################
        dhdq1 = terms[0].evaluate_derivative_q(phase_space) / npar
        prev_f1_sum = torch.sum(torch.square(-dhdq1), dim=-1)
        # prev_f1_sum.shape is [nsamples, nparticle]
        prev_f1_magnitude = torch.sqrt(prev_f1_sum)
        # prev_f1_magnitude.shape is [nsamples, nparticle]
        max_f1 = torch.max(prev_f1_magnitude)
        min_f1 = torch.min(prev_f1_magnitude)

        f1_magnitude, _ = torch.max(prev_f1_magnitude, dim=-1)
        # f1_magnitude.shape is [nsamples]

        check_f1 = -dhdqthrsh < f1_magnitude

        if check_f1.any() == True:  # HK
            s_idx = torch.where(check_f1)
            s_idx = torch.unique(s_idx[0], sorted=True)
            print('f1 too high, bb to garbage: ', f1_magnitude[s_idx], 'sample idx is ', s_idx)
            all_idx.append(s_idx)
            crash_flag = True

        check_p = pthrsh < abs(p_list)
	
        if check_p.any() == True:
            s_idx = torch.where(check_p)
            s_idx = torch.unique(s_idx[0], sorted=True)
            print('p too high: ',p_list[s_idx], 'sample idx is ', s_idx)
            all_idx.append(s_idx)
            crash_flag = True
            
        full_set = set(range(init_qp.shape[0]))

        if crash_flag == True:
            all_idx = torch.cat(all_idx)
            all_set = set(all_idx.tolist())
            diff_set = full_set - all_set
            diff_idx = list(diff_set)

            crsh_q = init_qp[:, 0, :, :][diff_idx]
            crsh_p = init_qp[:, 1, :, :][diff_idx]
            save_crash_config(crsh_q, crsh_p, boxsize, crash_path, rho[i])

        else:
            diff_idx = list(full_set)

        f1_magnitude = f1_magnitude[diff_idx]
        ##################### check min-max potential #######################
        # strike_append_u.shape = [nsamples]
        init_u = init_u[diff_idx]
        max_u = torch.max(init_u)
        min_u = torch.min(init_u)

        #################### check min-max momentum #####################
        # shape is [nsmaples, nparticle, DIM]
        p_list = p_list[diff_idx]
        # check min-max momentum
        # compare w pthrsh
        max_p = torch.max(p_list)
        min_p = torch.min(p_list)

        p_dist = p_list.reshape(-1)
        # shape is [nsmaples*nparticle*DIM]

        ##################### check min-max kinetic #######################
        ke = torch.sum(p_list * p_list / 2, dim=1)
        # sum along nparticle , ke shape is [nsamples, DIM]
        ke = torch.sum(ke, dim=1) / npar
        # sum along DIM , ke shape is [nsamples]
        # ke = ke / npar  # to compare kethresh. it is per particle.
        max_ke = torch.max(ke)
        min_ke = torch.min(ke)

        ##################### check min-max energy #######################
        init_e = init_e[diff_idx]

        max_e = torch.max(init_e)
        min_e = torch.min(init_e)

        ########################## print  ##############################
        print('rho', rho[i], 'ethrsh', ethrsh, 'e max', max_e.item(), 'min', min_e.item())
        print('rho', rho[i], 'kethrsh', kethrsh, 'ke max', max_ke.item(), 'min', min_ke.item())
        print('rho', rho[i], 'pthrsh', pthrsh, 'max_p', max_p.item(), 'min_p', min_p.item())
        print('rho', rho[i], 'pothrsh', pothrsh, 'u max', max_u.item(), 'min', min_u.item())
        print('rho', rho[i], '-dhdqthrsh', -dhdqthrsh, 'f1 max', max_f1.item(), 'min', min_f1.item())

        init_e1['e' + str(i + 1)] = init_e
        ke1['kinetic' + str(i + 1)] = ke
        p_dist1['pdist' + str(i + 1)] = p_dist
        init_u1['u' + str(i + 1)] = init_u
        f1_magnitude1['f' + str(i + 1)] =  f1_magnitude

    for i in range(len(filelist)):
        ##################### plot distribution #######################
        plt.hist((init_e1['e' + str(i + 1)] ).detach().numpy(), bins=100, alpha=.3, label= r'$\rho$={}'.format(rho[i]) )
        plt.xlabel('Energy', fontsize=20)
        plt.ylabel('hist', fontsize=20)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.legend(fontsize=17)
        plt.tight_layout()
    plt.show()

    for i in range(len(filelist)):
        plt.hist( ke1['kinetic' + str(i + 1)].detach().numpy(), bins=100, alpha=.3, label= r'$\rho$={}'.format(rho[i]) )
        plt.xlabel('Kinetic energy', fontsize=20)
        plt.ylabel('hist', fontsize=20)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.legend(fontsize=17)
        plt.tight_layout()
    plt.show()

    for i in range(len(filelist)):
        plt.hist(p_dist1['pdist' + str(i + 1)].detach().numpy(), bins=100, alpha=.3, label=r'$\rho$={}'.format(rho[i]) )
        plt.xlabel('Momentum', fontsize=20)
        plt.ylabel('hist', fontsize=20)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.legend(fontsize=17)
        plt.tight_layout()
    plt.show()

    for i in range(len(filelist)):
        plt.hist(init_u1['u' + str(i + 1)].detach().numpy(), bins=100, alpha=.3, label= r'$\rho$={}'.format(rho[i]))
        plt.xlabel('Potential energy', fontsize=20)
        plt.ylabel('hist', fontsize=20)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.legend(fontsize=17)
        plt.tight_layout()
    plt.show()

    for i in range(len(filelist)):
        plt.hist(f1_magnitude1['f' + str(i + 1)].detach().numpy(), bins=100, alpha=.3, label= r'$\rho$={}'.format(rho[i]))
        plt.xlabel('f1', fontsize=20)
        plt.ylabel('hist', fontsize=20)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.legend(fontsize=17)
        plt.tight_layout()
    plt.show()

