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
    def check(self, phase_space, prev_q, prev_p, prev_dHdq1, prev_dHdq2, prev_pred1, prev_pred2, hamiltonian):
        '''
        check if any samples have
        write crash configurations into crash file and then continue running excluded crash data
        returns None and continue running code if there is no crash configuration
        prev_q, prev_p, q_next, p_next shape is [nsamples, nparticles, DIM]
        prev_dHdq1, prev_dHdq2, prev_pred1, prev_pred2 shape is [nsamples, nparticles, DIM]
        hamiltonian: any hamiltonian object, valid for both noML or ML
        '''

        # new phase space
        q_next = phase_space.get_q()
        # shape [nsamples, nparticles, DIM]
        p_next = phase_space.get_p()
        boxsize = phase_space.get_boxsize()

        nparticle = q_next.shape[1]
        DIM = q_next.shape[2]

        next_energy_per_particle = hamiltonian.total_energy(phase_space) / nparticle
        # next_energy_per_particle.shape is [nsmaples]

        prev_dHdq1_per_particle = prev_dHdq1 / nparticle
        prev_dHdq2_per_particle = prev_dHdq2 / nparticle
        prev_pred1_per_particle = prev_pred1 / nparticle
        prev_pred2_per_particle = prev_pred2 / nparticle
        # shape is [nsamples, nparticle, (fx,fy)]

        all_idx = []
        crash_flag = False

        out_of_box = (torch.abs(q_next) > 0.5 * boxsize)  # check whether out of boundary

        if out_of_box.any() == True:
            # s_idx is the tuple, each index tensor contains indices for a certain dimension.
            s_idx = torch.where(out_of_box)  # condition (BoolTensor)

            # in s_idx, first index tensor represent indices for dim=0 that is along nsamples
            # remove duplicate values that are indices in s_idx
            s_idx = torch.unique(s_idx[0], sorted=True)
            print('q out of box error', 'sample idx is ', s_idx)
            # print to file and then quit
            all_idx.append(s_idx)
            crash_flag = True

        q_nan = torch.isnan(q_next); p_nan = torch.isnan(p_next) # check q or p is nan

        if (q_nan.any() or p_nan.any()) == True:
            s_idx = (torch.where(q_nan) or torch.where(p_next[p_nan]))
            s_idx = torch.unique(s_idx[0], sorted=True)
            print('q or p nan error', 'sample idx is ', s_idx)

            # print to file and then quit
            all_idx.append(s_idx)
            crash_flag = True

        # # check r thrsh
        # q_dim = q_next / boxsize
        # _, d = phase_space.paired_distance_reduced(q_dim, nparticle, DIM)
        # d = d * boxsize
        #
        # check_r =  d < self.rthrsh
        #
        # if check_r.any() == True:
        #     s_idx = torch.where(check_r)  # take sample index; tensor to int => s_idx[0].item()
        #     s_idx = torch.unique(s_idx[0], sorted=True)
        #     print('check_r more than rthrsh: ', d[s_idx],  'sample idx is ', s_idx)
        #     all_idx.append(s_idx)
        #     crash_flag = True

        # check energy per par more than ethrsh and less than ethrsh * 1.5
        # check_e =  next_energy_per_particle > self.ethrsh
        #
        # if check_e.any() == True:
        #     s_idx = torch.where(check_e)  # take sample index; tensor to int => s_idx[0].item()
        #     s_idx = torch.unique(s_idx[0], sorted=True)
        #     print('energy_per_particle too high: ', next_energy_per_particle[s_idx],  'sample idx is ', s_idx)
        #     all_idx.append(s_idx)
        #     crash_flag = True
        #
        # # check p next is more than pthrsh and less than pthrsh * 1.2
        # check_p = abs(p_next) > self.pthrsh
        #
        # if check_p.any() == True:
        #     s_idx = torch.where(check_p)
        #     s_idx = torch.unique(s_idx[0], sorted=True)
        #     print('p too high: ', p_next[s_idx], 'sample idx is ', s_idx)
        #     all_idx.append(s_idx)
        #     crash_flag = True

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

            self.save_crash_config(crsh_q, crsh_p, boxsize, self.crash_path)
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

            phase_space.set_q(diff_q)
            phase_space.set_p(diff_p)

            return crsh_set

    # ======================================
    def save_crash_config(self, crash_q_list, crash_p_list, boxsize, crash_path):
        ''' log crash state in a new file every time crashed '''

        now = datetime.now()
        dt_string = now.strftime("%Y%m%d%H%M%S")

        if not os.path.exists(crash_path): 
            os.makedirs(crash_path)

        crash_filename = crash_path + 'crashed_at' + dt_string + '_' + str(self.crash_id) + '.pt'

        # write q, p list
        qp_list = torch.stack((crash_q_list, crash_p_list), dim=1)
        qp_list = torch.unsqueeze(qp_list, dim=2)
        data_io.write_trajectory_qp(crash_filename, qp_list, boxsize)
        # shape is [crashed nsamples, (q,p), 1, npaticle, DIM]


