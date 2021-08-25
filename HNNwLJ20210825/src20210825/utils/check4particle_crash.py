import torch
from datetime import datetime
from utils.data_io import data_io
import os


class check4particle_crash:
    '''  this class  use for check debug '''

    def __init__(self, rthrsh0, pthrsh0, rthrsh, pthrsh, crash_path):
        '''
        [rthrsh0, rthrsh2] : threshold for pair of particles
        [pthrsh0, pthrsh2] : threshold for momentum
        [kethrsh, kethrsh2] : threshold for ke per particle
        [pothrsh, pothrsh2] : threshold for po per particle
        [ethrsh, ethrsh2] : threshold for total energy per particle - cannot be too high
        [dhdqthrsh, dhdqthrsh2] : threshold for dhdq per particle
        boxsize: size of 2D box for LJ
        crash_path : save data before crash to this path
        crash_id : count when crash
        '''

        self.crash_path = crash_path

        self.rthrsh0 = rthrsh0
        self.pthrsh0 = pthrsh0

        self.pthrsh = pthrsh
        self.alpha12 = pow(2, 1/12) # alpha = 2

        rthrsh2 = rthrsh / self.alpha12
        self.pthrsh2 = self.pthrsh * self.alpha12

        kethrsh = pthrsh0 * pthrsh0 / 2 + pthrsh0 * pthrsh0 / 2
        kethrsh2 = self.pthrsh2 * self.pthrsh2 / 2 + self.pthrsh2 * self.pthrsh2 / 2

        pothrsh = 4 * (1 / (rthrsh0) ** 12 - 1 / (rthrsh0) ** 6)
        pothrsh2 = 4 * (1 / (rthrsh2) ** 12 - 1 / (rthrsh2) ** 6)

        self.ethrsh = kethrsh + pothrsh
        self.ethrsh2 = kethrsh2 + pothrsh2

        self.dhdqthrsh = 4 * ((-12) / (rthrsh0) ** 13)  - 4 * ((-6) / (rthrsh0) ** 7)
        self.dhdqthrsh2 = 4 * ((-12) / (rthrsh2) ** 13) - 4 * ((-6) / (rthrsh2) ** 7)
        self.crash_id = 0

        print('check4particle_crash initialized : rthrsh0', rthrsh0, 'rthrsh2 ', rthrsh2, 'ethrsh ',self.ethrsh,
              'ethrsh2 ',self.ethrsh2, ' pthrsh0 ',pthrsh0, ' pthrsh2 ',self.pthrsh2, 'dhdqthrsh ',self.dhdqthrsh,
              'dhdqthrsh2 ',self.dhdqthrsh2)


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
        p_next = phase_space.get_p()
        boxsize = phase_space.get_boxsize()

        nparticle = q_next.shape[1]

        next_energy_per_particle = hamiltonian.total_energy(phase_space) / nparticle
        # next_energy_per_particle.shape is [nsmaples]

        all_idx = []
        gar_idx = []
        crash_flag = False
        garbage_flag = False
        garbage_set = None # default None if no garbage set

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

        # check energy per par more than ethrsh and less than ethrsh * 1.5
        check_e = (next_energy_per_particle > self.ethrsh) & (self.ethrsh2 > next_energy_per_particle)

        if check_e.any() == True:
            s_idx = torch.where(check_e)  # take sample index; tensor to int => s_idx[0].item()
            s_idx = torch.unique(s_idx[0], sorted=True)
            print('energy_per_particle too high: ', next_energy_per_particle[s_idx], 'sample idx is ', s_idx)
            all_idx.append(s_idx)
            crash_flag = True

        garbage_e = next_energy_per_particle >= self.ethrsh2

        if garbage_e.any() == True:
            s_idx = torch.where(garbage_e)
            s_idx = torch.unique(s_idx[0], sorted=True)
            print('energy_per_particle too high, bb to garbage: ', next_energy_per_particle[s_idx], 'sample idx is ', s_idx)
            gar_idx.append(s_idx)
            garbage_flag = True

        # check p next is more than pthrsh and less than pthrsh * 1.2
        check_p = (abs(p_next) > self.pthrsh0) & (self.pthrsh2 > abs(p_next))

        if check_p.any() == True:
            s_idx = torch.where(check_p)
            s_idx = torch.unique(s_idx[0], sorted=True)
            print('p too high: ',p_next[s_idx], 'sample idx is ', s_idx)
            all_idx.append(s_idx)
            crash_flag = True

        garbage_p = abs(p_next) >= self.pthrsh2

        if garbage_p.any() == True:
            s_idx = torch.where(garbage_p)
            s_idx = torch.unique(s_idx[0], sorted=True)
            print('p too high, bb to garbage: ', p_next[s_idx], 'sample idx is ', s_idx)
            gar_idx.append(s_idx)
            garbage_flag = True

        prev_f1 = - prev_dHdq1 - prev_pred1
        # prev_f1.shape is [nsamples, nparticle, (fx,fy)]

        prev_f1_sum = torch.sum(torch.square(prev_f1), dim=-1)
        # prev_f1_sum.shape is [nsamples, nparticle]

        prev_f1_magnitude = torch.sqrt(prev_f1_sum)
        # prev_f1_magnitude.shape is [nsamples, nparticle]

        try:
            f1_magnitude, _ = torch.max(prev_f1_magnitude, dim=-1)
            # f1_magnitude.shape is [nsamples]
            # force per particle that has maximum force

            check_f1 = (f1_magnitude > -self.dhdqthrsh) & (-self.dhdqthrsh2 > f1_magnitude)
            # check force 1 magnitude more than absolute dhdqthrsh and less than absolute dhdqthrsh * 1.5

            if check_f1.any() == True:

                s_idx = torch.where(check_f1)
                s_idx = torch.unique(s_idx[0], sorted=True)
                print('f1 too high: ', f1_magnitude[s_idx], 'sample idx is ', s_idx)
                all_idx.append(s_idx)
                crash_flag = True

            garbage_f1 = f1_magnitude >= -self.dhdqthrsh2

            if garbage_f1.any() == True:  # HK
                s_idx = torch.where(garbage_f1)
                s_idx = torch.unique(s_idx[0], sorted=True)
                print('f1 too high, bb to garbage: ', f1_magnitude[s_idx], 'sample idx is ', s_idx)
                gar_idx.append(s_idx)
                garbage_flag = True

        except Exception:
            print('pass ... all samples crash ... cannot perform max on tensor with no elements')
            pass

        prev_f2 = - prev_dHdq2 - prev_pred2
        # prev_f2.shape is [nsamples, nparticle, (fx,fy)]

        prev_f2_sum = torch.sum(torch.square(prev_f2), dim=-1)
        # prev_f2_sum.shape is [nsamples, nparticle]

        prev_f2_magnitude = torch.sqrt(prev_f2_sum)
        # prev_f2_magnitude.shape is [nsamples, nparticle]

        try:
            f2_magnitude, _ = torch.max(prev_f2_magnitude, dim=-1)
            # f2_magnitude.shape is [nsamples]
            # force per particle that has maximum force

            check_f2 = (f2_magnitude > -self.dhdqthrsh) & (-self.dhdqthrsh2 > f2_magnitude)
            # check force 2 magnitude more than absolute dhdqthrsh and less than absolute dhdqthrsh * 1.5

            if check_f2.any() == True:

                s_idx = torch.where(check_f2)
                s_idx = torch.unique(s_idx[0], sorted=True)
                print('f2 too high: ', f2_magnitude[s_idx], 'sample idx is ', s_idx)
                all_idx.append(s_idx)
                crash_flag = True

            garbage_f2 = f2_magnitude >= -self.dhdqthrsh2

            if garbage_f2.any() == True:
                s_idx = torch.where(garbage_f2)
                s_idx = torch.unique(s_idx[0], sorted=True)
                print('f2 too high, bb to garbage : ', f2_magnitude[s_idx], 'sample idx is ', s_idx)
                gar_idx.append(s_idx)
                garbage_flag = True

        except Exception:
            print('pass ... all samples crash ... cannot perform max on tensor with no elements')
            pass

        if garbage_flag == True:

            # collect indices in garbage data that give out of thrsh condition ( e.g. more than ethrsh * 1.2  )
            # and not use these garbage data for retraining model
            gar_idx = torch.cat(gar_idx)
            gar_idx = torch.unique(gar_idx)

            gar_q = prev_q[gar_idx]
            gar_p = prev_p[gar_idx]

            self.save_crash_config(gar_q, gar_p, boxsize, self.crash_path + 'garbages/')

            garbage_set = set(gar_idx.tolist())
            print('garbage set', len(garbage_set))

        if crash_flag == True:

            self.crash_id += 1

            # all_set is indices of all crash data = crsh data + garbage data
            # crsh data : give soft thrsh condition as possible
            all_idx = torch.cat(all_idx)
            all_idx = torch.unique(all_idx)

            all_set = set(all_idx.tolist())

            if garbage_set is not None:
                # crsh set is subtract garbage set from all crash set to remove same garbage indices
                crsh_set = all_set - garbage_set
                print('crsh set that subtract garbage set', len(crsh_set))
                crsh_idx = list(crsh_set)

            else:
                crsh_set = all_set
                print('crsh set that is garbage set None', len(crsh_set))
                crsh_idx = list(crsh_set)

            crsh_q = prev_q[crsh_idx] # take pre_q that the next_q is crash
            crsh_p = prev_p[crsh_idx]

            self.save_crash_config(crsh_q, crsh_p, boxsize, self.crash_path)
            # print('saving qp_state_before_crash for ',all_idx)

            # reduce nsamples by remove all crash data
            self.rm_crsh_samples(all_set, garbage_set, q_next, p_next, phase_space)

            return crsh_set
            # ======================================

    def rm_crsh_samples(self, all_set, garbage_set, q_next, p_next, phase_space):

        # all samples before remove crash data
        full_set = set(range(q_next.shape[0]))
        print('full set', len(full_set))

        if garbage_set is not None:
            # remove crash data and garbage data
            diff_set = full_set - all_set - garbage_set
            print('diff set',len(diff_set))
        else:
            # remove only crash data
            diff_set = full_set - all_set
            print('diff set',len(diff_set))

        diff_list = list(diff_set)
        diff_q = q_next[diff_list]
        diff_p = p_next[diff_list]  # here remove crash samples

        phase_space.set_q(diff_q)
        phase_space.set_p(diff_p)
        
        #if len(diff_set) == 0 :
        #    print('diff set 0 ..... quit ')
        #    quit()        

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


