import torch

class linear_integrator:

    ''' linear_integrator class to help implement numerical integrator at each step '''

    _obj_count = 0

    def __init__(self, integrator_method, crash_chker):

        '''
        parameters
        ___________
        crash_chker : check and collect crash data
        use crash checker dummy when prepare labels and training
        else use predict trajectory till no crash data
        '''

        linear_integrator._obj_count += 1
        assert (linear_integrator._obj_count <= 2),type(self).__name__ + " has more than one object"

        self._integrator_method = integrator_method

        self.crash_checker = crash_chker
        print('linear_integrator initialized ')

    # =================================================

    def one_step(self, hamiltonian, phase_space, tau_cur):

        ''' do one step of time integration and check crash using crash_checker
        when predict long trajectory to collect crash data
        otherwise use crash checker dummy (prepare label, training, gold standard)

        Parameters
        ----------
        hamiltonian : can be ML or noML hamiltonian
        phase_space : contains q_list, p_list as input for integration and contains boxsize also
        tau_cur : float
                large time step for prediction
                short time step for label
        crash_idx : indices of crashed qp_list
                if crash, return crash indices. otherwise None

        return :torch.tensor
                shape is [nsamples, (q, p), nparticle, DIM]
        '''

        prev_q = phase_space.get_q()
        prev_p = phase_space.get_p()

        boxsize = phase_space.get_boxsize()
        q_next, p_next, dhdq1, dhdq2, pred1, pred2  = self._integrator_method(hamiltonian, phase_space, tau_cur, boxsize)
        # all shape is [nsamples, nparticle, DIM]

        # print('q, p shape before crash checker ', q_list.shape, p_list.shape)

        crash_idx = self.crash_checker.check(phase_space, prev_q, prev_p, dhdq1, dhdq2, pred1, pred2, hamiltonian)
        #print('q, p shape after crash checker ',phase_space.get_q().shape, phase_space.get_p().shape, 'crash_idx ', crash_idx)

        qp_list = torch.stack((phase_space.get_q(), phase_space.get_p()), dim=1)
        # shape is [nsamples, (q,p), nparticle, DIM]

        return qp_list, crash_idx

    # =====================================================

    def nsteps(self, hamiltonian, phase_space, tau_cur, nitr, append_strike):

        ''' to integrate more than one step

        nitr : number of strike steps to save file for MD
        append_strike : the period of which qp_list append

        return :
        qp_list : list
                append nxt_qp to the qp list
                if get crash, make qp_list empty
        crash_flag : True or False
                if crash_idx exist, return crash_flag is True 
        '''

        assert (nitr % append_strike == 0), 'incompatible strike and nitr'

        crash_flag = False

        qp_list = [] # list to append qp_list. qp_list become empty if some samples have crash
        crash_iter = [] # list to append iteration that get crash
        crash_ct = [] # list to append the number of crash data at iteration that get crash

        for t in range(nitr):
            print('====== step ', t, flush=True)
            nxt_qp, crash_idx = self.one_step( hamiltonian, phase_space, tau_cur)
            # nxt_qp.shape is [nsamples, (q, p), nparticle, DIM]

            if (t+1) % append_strike == 0: # append_strike = 1

                # if crash_idx is not None, make empty qp_list and crash_flag is True
                # append iteration of crash data and the number of crash data
                if crash_idx is not None:
                    qp_list = []
                    crash_iter.append(t)
                    crash_ct.append(len(crash_idx))
                    crash_flag = True

                qp_list.append(nxt_qp)
        
        return qp_list, crash_flag, crash_iter, crash_ct

    def rm_crash_nsteps(self, hamiltonian, phase_space, tau_cur, nitr):

        ''' to integrate more than one step and remove samples that have crash

        nitr : number of strike steps to save file for MD

        return :
        qp_list : torch.tensor
                concat nxt_qp to nxt_qp
        '''

        # need initial state to know which samples that don't have crash
        # use these samples to analysize the distance matries
        init_qp = torch.stack((phase_space.get_q(), phase_space.get_p()), dim=1)

        qp_list = None

        for t in range(nitr):
            print('====== step ', t, flush=True)

            # use prev q index to remove crashed nsamples
            prev_q = phase_space.get_q()
            full_set = set(range(prev_q.shape[0]))

            # if crsh set exist, get reduced nsamples after remove samples that have crash
            nxt_qp, crsh_set = self.one_step(hamiltonian, phase_space, tau_cur)
            # nxt_qp.shape is [nsamples, (q, p), nparticle, DIM]

            # use diff set to remove crashed nsamples
            if crsh_set is not None:
                diff_set = full_set - crsh_set
            else:
                diff_set = full_set

            diff_list = list(diff_set)

            if t == 0:
                if crsh_set is not None:
                    # take no crash index from initial state and then stack with nxt state
                    qp_list = torch.stack((init_qp[diff_list], nxt_qp),dim=2)
                    # qp_list.shape is [nsamples, (q, p), trajectory length, nparticle, DIM]

                else:
                    # take index from inital state and then stack with nxt state
                    qp_list = torch.stack((init_qp, nxt_qp), dim=2)

            else:
                if crsh_set is not None:
                    nxt_qp = torch.unsqueeze(nxt_qp, dim=2)
                    # nxt_qp.shape is [nsamples, (q, p), 1, nparticle, DIM]

                    # take no crash index from qp_list and then stack with nxt_qp
                    qp_list = torch.cat((qp_list[diff_list],nxt_qp),dim=2)

                else:
                    nxt_qp = torch.unsqueeze(nxt_qp, dim=2)
                    # nxt_qp.shape is [nsamples, (q, p), 1, nparticle, DIM]

                    # no crash index from qp_list and then stack with nxt_qp
                    qp_list = torch.cat((qp_list, nxt_qp), dim=2)

        return qp_list
