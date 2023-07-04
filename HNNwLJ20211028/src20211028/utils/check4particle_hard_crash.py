import torch
from datetime import datetime
from utils.data_io import data_io
import os


class check4particle_hard_crash:
    '''  this class  use for check debug '''

    def __init__(self, rthrsh, pthrsh, crash_path):
        '''
        rthrsh : threshold for pair of particles
        pthrsh : threshold for momentum
        kethrsh : threshold for ke per particle
        pothrsh : threshold for po per particle
        ethrsh : threshold for total energy per particle - cannot be too high
        dhdqthrsh : threshold for dhdq per particle
        boxsize: size of 2D box for LJ
        crash_path : save data before crash to this path
        crash_id : count when crash
        '''

        self.crash_path = crash_path
        self.pthrsh = pthrsh
        self.rthrsh = rthrsh

        kethrsh = pthrsh * pthrsh / 2 + pthrsh * pthrsh / 2
        pothrsh = 4 * (1 / (self.rthrsh) ** 12 - 1 / (self.rthrsh) ** 6)

        self.ethrsh = kethrsh + pothrsh

        self.dhdqthrsh = 4 * ((-12) / (rthrsh) ** 13)  - 4 * ((-6) / (self.rthrsh) ** 7)
        self.crash_id = 0

        print('check4particle_hard_crash initialized : rthrsh', rthrsh, ' pthrsh ', pthrsh,
              'ethrsh ',self.ethrsh, 'dhdqthrsh ',self.dhdqthrsh)

    # ============================================
    def check(self, phase_space, prev_q, prev_p, prev_l, prev_dHdq1, prev_dHdq2, prev_pred1, prev_pred2, hamiltonian):
        '''
        check if any samples have
        write crash configurations into crash file and then continue running excluded crash data
        returns None and continue running code if there is no crash configuration
        prev_q, prev_p, prev_bs, prev_l, q_next, p_next shape is [nsamples, nparticles, DIM]
        prev_dHdq1, prev_dHdq2, prev_pred1, prev_pred2 shape is [nsamples, nparticles, DIM]
        hamiltonian: any hamiltonian object, valid for both noML or ML
        '''

        # new phase space
        q_next = phase_space.get_q()
        p_next = phase_space.get_p()
        l_next = phase_space.get_l_list()
        # shape [nsamples, nparticles, DIM]

        nparticle = q_next.shape[1]

        prev_dHdq1_per_particle = prev_dHdq1 / nparticle
        prev_dHdq2_per_particle = prev_dHdq2 / nparticle
        prev_pred1_per_particle = prev_pred1 / nparticle
        prev_pred2_per_particle = prev_pred2 / nparticle
        # shape is [nsamples, nparticle, (fx,fy)]

        all_idx = []
        crash_flag = False

        q_nan = torch.isnan(q_next); p_nan = torch.isnan(p_next) # check q or p is nan

        if (q_nan.any() or p_nan.any()) == True:
            s_idx = (torch.where(q_nan) or torch.where(p_next[p_nan]))
            s_idx = torch.unique(s_idx[0], sorted=True)
            print('q or p nan error', 'sample idx is ', s_idx)

            # print to file and then quit
            all_idx.append(s_idx)
            crash_flag = True


        prev_f1 = - prev_dHdq1_per_particle - prev_pred1_per_particle
        # prev_f1.shape is [nsamples, nparticle, (fx,fy)]

        prev_f1_sum = torch.sum(torch.square(prev_f1), dim=-1)
        # prev_f1_sum.shape is [nsamples, nparticle]

        prev_f1_magnitude = torch.sqrt(prev_f1_sum)
        # prev_f1_magnitude.shape is [nsamples, nparticle]

        try:
            f1_magnitude, _ = torch.max(prev_f1_magnitude, dim=-1)
            # f1_magnitude.shape is [nsamples]
            # force per particle that has maximum force

            check_f1 =  f1_magnitude > -self.dhdqthrsh
            # check force 1 magnitude more than - dhdqthrsh

            if check_f1.any() == True:

                s_idx = torch.where(check_f1)
                s_idx = torch.unique(s_idx[0], sorted=True)
                print('f1 too high: ', f1_magnitude[s_idx], 'sample idx is ', s_idx)
                all_idx.append(s_idx)
                crash_flag = True

        except Exception:
            print('pass ... all samples crash ... cannot perform max on tensor with no elements')
            pass


        prev_f2 = - prev_dHdq2_per_particle - prev_pred2_per_particle
        # prev_f2.shape is [nsamples, nparticle, (fx,fy)]

        prev_f2_sum = torch.sum(torch.square(prev_f2), dim=-1)
        # prev_f2_sum.shape is [nsamples, nparticle]

        prev_f2_magnitude = torch.sqrt(prev_f2_sum)
        # prev_f2_magnitude.shape is [nsamples, nparticle]

        try:
            f2_magnitude, _ = torch.max(prev_f2_magnitude, dim=-1)
            # f2_magnitude.shape is [nsamples]
            # force per particle that has maximum force

            check_f2 = f2_magnitude > -self.dhdqthrsh
            # check force 2 magnitude more than - dhdqthrsh

            if check_f2.any() == True:

                s_idx = torch.where(check_f2)
                s_idx = torch.unique(s_idx[0], sorted=True)
                print('f2 too high: ', f2_magnitude[s_idx], 'sample idx is ', s_idx)
                all_idx.append(s_idx)
                crash_flag = True

        except Exception:
            print('pass ... all samples crash ... cannot perform max on tensor with no elements')
            pass


        if crash_flag == True:

            self.crash_id += 1

            all_idx = torch.cat(all_idx)
            all_idx = torch.unique(all_idx)

            crsh_q = prev_q[all_idx] # take pre_q that the next_q is crash
            crsh_p = prev_p[all_idx]
            crsh_l = prev_l[all_idx]

            self.save_crash_config(crsh_q, crsh_p, crsh_l, self.crash_path)
            # print('saving qp_state_before_crash for ',all_idx)

            full_set = set(range(q_next.shape[0]))
            print('full', len(full_set))
            crsh_set = set(all_idx.tolist())
            print('crsh', len(all_idx))
            diff_set = full_set - crsh_set
            print('diff', len(diff_set))

            diff_list = list(diff_set)
            diff_q = q_next[diff_list]
            diff_p = p_next[diff_list]  # here remove crash samples
            diff_l = l_next[diff_list]

            phase_space.set_q(diff_q)
            phase_space.set_p(diff_p)
            phase_space.set_l_list(diff_l)

            return crsh_set

    # ======================================
    def save_crash_config(self, crash_q_list, crash_p_list, crash_l_list, crash_path):
        ''' log crash state in a new file every time crashed '''

        now = datetime.now()
        dt_string = now.strftime("%Y%m%d%H%M%S")

        if not os.path.exists(crash_path): 
            os.makedirs(crash_path)

        crash_filename = crash_path + 'crashed_at' + dt_string + '_' + str(self.crash_id) + '.pt'

        # write q, p, boxsize list
        qpl_list = torch.stack((crash_q_list, crash_p_list, crash_l_list), dim=1)
        # shape is [crashed nsamples, (q,p,boxsize), npaticle, DIM]

        qpl_list = torch.unsqueeze(qpl_list, dim=2)
        data_io.write_trajectory_qpl(crash_filename, qpl_list)
        # qpl_list.shape is [crashed nsamples, (q,p,boxsize), 1, npaticle, DIM]


