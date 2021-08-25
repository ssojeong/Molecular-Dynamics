import torch
import math
import sys
import json
import matplotlib.pyplot as plt


def del_qp(traj_st, traj_lt):

    nsamples, _, _, nparticle, _ = traj_st.shape

    del_q = torch.pow((traj_st[:, 0, :, :, :] - traj_lt[:, 0, :, :, :]), 2)
    del_p = torch.pow((traj_st[:, 1, :, :, :] - traj_lt[:, 1, :, :, :]), 2)
    # shape = [nsamples, trajectory, nparticles, DIM]

    avg_del_q_particle = torch.sum(del_q, dim=2) / nparticle
    avg_del_p_particle = torch.sum(del_p, dim=2) / nparticle
    # shape = [nsamples, trajectory, DIM]

    avg_del_q_particle = torch.sum(avg_del_q_particle, dim=2)
    avg_del_p_particle = torch.sum(avg_del_p_particle, dim=2)
    # shape = [nsamples, trajectory]

    del_qp_particle = avg_del_q_particle + avg_del_p_particle
    # shape = [nsamples, trajectory]

    avg2_del_qp_particle_sample = torch.sum(torch.pow(del_qp_particle, 2), dim=0) / nsamples

    avg_del_q_particle_sample = torch.sum(avg_del_q_particle, dim=0) / nsamples
    # shape = [trajectory]

    avg_del_p_particle_sample = torch.sum(avg_del_p_particle, dim=0) / nsamples
    # shape = [trajectory]

    avg_del_qp_particle_sample = avg_del_q_particle_sample + avg_del_p_particle_sample

    std_del_qp_particle_sample = (avg2_del_qp_particle_sample - avg_del_qp_particle_sample * avg_del_qp_particle_sample) ** 0.5

    return avg_del_q_particle_sample, avg_del_p_particle_sample, avg_del_qp_particle_sample


if __name__ == '__main__':

    argv = sys.argv
    json_file = argv[1]

    with open(json_file) as f:
        data = json.load(f)

    file_st = data['file_st']
    file_lt = data['file_lt']
    filenames = data['filenames']
    color = data['color']
    MLornoML = data['MLornoML']

    max_ts_cut = 100

    collected_samples = {}
    avg_del_q = {}
    avg_del_p = {}
    avg_del_qp = {}

    for i in range(len(file_st)):

        info_st = torch.load(file_st[i])
        info_lt = torch.load(file_lt[i])

        nsamples, _, iter, npar, DIM = info_st['qp_trajectory'].shape

        traj_st = info_st['qp_trajectory']
        traj_lt = info_lt['qp_trajectory']

        if MLornoML[i] == 'yes':
            traj_lt = traj_lt[:, :, 1:, :, :]
        else:
            traj_lt = traj_lt[:, :, :, :, :]

        collected_samples['nsamples' + str(i+1)] = nsamples

        avg_del_q['del_q' + str(i + 1)], avg_del_p['del_p' + str(i + 1)], avg_del_qp['del_qp' + str(i + 1)] = del_qp(
            traj_st[:,:,:,:,:], traj_lt)

        tau_short = info_lt['tau_short']
        tau_long = info_lt['tau_long']
        boxsize  = info_lt['boxsize']


    pair_time_step = 0.1
    L_h = boxsize/2.
    q_max = math.sqrt(L_h*L_h + L_h*L_h)
    print('Maximum distance r = {}, r^2 = {}'.format(q_max,q_max*q_max))

    fig = plt.figure()
    t = torch.arange(1., max_ts_cut+1)

    plt.suptitle(
        r'nparticle {}, boxsize {:.4f}, Maximum distance $r = {:.4f}, r^2 = {:.4f}$'.format(npar, boxsize,
                                                                                            q_max,
                                                                                            q_max * q_max)
        + '\n pair with time step = {}, Maximum time step = {}, '.format(pair_time_step, max_ts_cut)
        + r'$\tau^\prime$ = {}'.format(tau_short))

    for i in range(len(file_st)):

        fig.add_subplot(2,2,1)
        plt.ylabel(r'$\Delta q^{\tau,{\tau}^\prime}$',fontsize=15)
        plt.plot(t,avg_del_q['del_q' +str(i+1)][:max_ts_cut], color[i], label = filenames[i])
        plt.ylim(-0.05, 16)
        plt.grid()
        plt.legend()

        fig.add_subplot(2,2,2)
        plt.ylabel(r'$\Delta p^{\tau,{\tau}^\prime}$',fontsize=15)
        plt.plot(t,avg_del_p['del_p' +str(i+1)][:max_ts_cut], color[i], label = filenames[i])
        plt.ylim(-0.05, 5)
        plt.grid()
        plt.legend()

        fig.add_subplot(2,1,2)
        plt.xlabel('time',fontsize=16)
        plt.ylabel(r'$\Delta^{\tau,{\tau}^\prime}$',fontsize=18)
        plt.plot(t,avg_del_qp['del_qp' +str(i+1)][:max_ts_cut], color[i], label = 'nsamples{}'.format(collected_samples['nsamples' + str(i+1)]))
        plt.ylim(-0.05,17)
        plt.tick_params(axis='y',labelsize=16)
        plt.grid()
        plt.legend()

    plt.show()
