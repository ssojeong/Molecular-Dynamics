from parameters.MD_parameters               import MD_parameters
from parameters.ML_parameters               import ML_parameters
from phase_space                            import phase_space
from integrator.linear_integrator           import linear_integrator
from utils.data_io                          import data_io
from utils.make_hamiltonian                 import make_hamiltonian
from HNN.checkpoint                         import checkpoint
from utils.check4particle_soft_crash        import check4particle_soft_crash
from utils.check4particle_hard_crash        import check4particle_hard_crash
from utils.check4particle_crash_dummy       import check4particle_crash_dummy  # crash data prepare label

import sys
import time
import math
import psutil
import shutil
import torch

if __name__=='__main__':
    # need to load ML json file to get ML paramters
    # run something like this
    # python MD_sampler.py ../data/gen_by_MD/n16rho0.2/n16T0.27seed1299nsamples5/MD_config.dict ../data/gen_by_MD/n16rho0.2/n16T0.27seed1299nsamples5/ML_config.dict

    argv = sys.argv
    MDjson_file = argv[1]
    MLjson_file = argv[2]

    MD_parameters.load_dict(MDjson_file)
    ML_parameters.load_dict(MLjson_file)

    seed = ML_parameters.seed
    torch.manual_seed(seed)

    torch.set_default_dtype(torch.float64)

    tau_short        = MD_parameters.tau_short
    append_strike    = MD_parameters.append_strike
    save2file_strike = MD_parameters.save2file_strike
    tau_long         = MD_parameters.tau_long
    niter_tau_short  = MD_parameters.niter_tau_short
    
    long_traj_save2file = MD_parameters.long_traj_save2file
    long_traj_no_crash_save = MD_parameters.long_traj_no_crash_save

    # io varaiables
    MC_init_config_filename       = MD_parameters.MC_init_config_filename
    MD_output_basenames           = MD_parameters.MD_output_basenames
    crash_filename                = MD_parameters.crash_filename  # noML : None, otherwise have filename
    crash_path                    = MD_parameters.crash_path      # noML : None, otherwise have path
    crash_checker                 = MD_parameters.crash_checker   # noML : 'no', otherwise 'yes'
    crash_thrsh                   = MD_parameters.crash_thrsh     # yes: soft carsh, no: hard crash
    print('MC output filename', MC_init_config_filename)

    # crash checker variables
    rthrsh0                       = MD_parameters.rthrsh0
    pthrsh0                       = MD_parameters.pthrsh0  
    rthrsh                        = MD_parameters.rthrsh
    pthrsh                        = MD_parameters.pthrsh   # T = 1.0 is given

    # MD variables
    tau_cur          = tau_short
    tau_short        = tau_short
    tau_long         = tau_long
    
    hamiltonian_type = MD_parameters.hamiltonian_type 
    n_out_files      = niter_tau_short // save2file_strike
    print('tau cur', tau_cur ) 

    # ML variables
    load_model_file  = ML_parameters.ML_chk_pt_filename

    phase_space = phase_space.phase_space()

    if crash_checker == 'yes':
        if crash_thrsh == 'yes':
            crsh_chker = check4particle_soft_crash(rthrsh0, pthrsh0, rthrsh, pthrsh, crash_path)
        else :
            crsh_chker = check4particle_hard_crash(rthrsh, pthrsh, crash_path)
    else:
        crsh_chker = check4particle_crash_dummy(rthrsh0, pthrsh0, rthrsh, pthrsh, crash_path)

    linear_integrator_obj = linear_integrator( MD_parameters.integrator_method, crsh_chker )

    hamiltonian_obj = make_hamiltonian(hamiltonian_type, tau_long, ML_parameters)

    if hamiltonian_type != "noML": # use crash
        chk_pt = checkpoint(hamiltonian_obj.get_netlist()) # opt = None, sch = None ; for test, don't need opt, sch
        if load_model_file is not None: chk_pt.load_checkpoint(load_model_file)
        hamiltonian_obj.eval()
        hamiltonian_obj.requires_grad_false()

    init_qp, _, _, boxsize = data_io.read_trajectory_qp(MC_init_config_filename)
    # init_qp.shape = [nsamples, (q, p), 1, nparticle, DIM]

    init_q = torch.squeeze(init_qp[:,0,0,:,:], dim=1)
    # init_q.shape = [nsamples, nparticle, DIM]

    init_p = torch.squeeze(init_qp[:,1,0,:,:], dim=1)
    # init_p.shape = [nsamples, nparticle, DIM]

    phase_space.set_q(init_q)
    phase_space.set_p(init_p)
    phase_space.set_boxsize(boxsize)

    crash_history = False

    print('===== start time to save file or collect crash ... ')
    start = time.time()
    # write file
    for i in range(n_out_files):

        if long_traj_no_crash_save == 'yes':
        # use to prepare label ; save trajectory with no care crash

            qp_list, crash_flag, crash_iter, crash_ct = linear_integrator_obj.nsteps(hamiltonian_obj, phase_space, tau_cur,
                                                                                     save2file_strike, append_strike)
            # qp_list is [nsamples, (q, p), nparticle, DIM] append to trajectory length number of items
            # crash_flag is true, some of nsamples have crashed qp_list
            # crash_flag is false, nsamples don't have crashed qp_list
            # crash_iter, crash_ct is for plot of histogram about crash data

            crash_history = crash_flag

            # if crash_flag is false, write file to save qp list in intermediate steps of integration
            if crash_history is False:

                qp_list = torch.stack(qp_list, dim=2)
                # qp_list.shape = [nsamples, (q, p), trajectory length, nparticle, DIM]

                print('no crash created file i', i, 'memory % used:', psutil.virtual_memory()[2], '\n')

                if long_traj_save2file == 'yes':
                    print('so that save file idx', i)
                    tmp_filename = MD_output_basenames + '_id' + str(i) + '.pt'
                    data_io.write_trajectory_qp(tmp_filename, qp_list, boxsize, tau_short, tau_long)

                else:
                    print('not save file .... large storage... only collect crash data....')

            else: # to plot crash histogram
                print('collect iter that get crash..')
                # To continue to iterate every index of n_out_files, sum iter and index * save2file_strike
                crash_niter = torch.tensor(crash_iter) + save2file_strike * i
                crash_nct = torch.tensor(crash_ct)

                data_io.write_crash_info(crash_filename + '_id' + str(i) + '.pt', crash_niter.long(), crash_nct.long())
                print('crash iter', crash_niter, 'count', crash_nct)
                print('write file for crash info id ', i)

            if qp_list[0].shape[0] == 0:
                print('tensor of 0 elements .... mean all samples crash .... quit running')
                quit()

        else:
            # write file to save qp list that "remove crash data" in intermediate steps of integration
            print('rm crash and save file idx', i)
            qp_list = linear_integrator_obj.rm_crash_nsteps(hamiltonian_obj, phase_space, tau_cur, save2file_strike)
            print('qp list shape', qp_list.shape)
            _filename = MD_output_basenames + '_metric.pt'
            data_io.write_trajectory_qp(_filename, qp_list, boxsize, tau_short, tau_long)

    end = time.time()
    print('===== end time to save file or collect crash ... {:.3f}sec'.format(end - start))

    # cp init file in same training filename folder
    shutil.copy2(MC_init_config_filename, MD_parameters.MD_data_dir)
    print('file write dir:', MD_parameters.MD_data_dir)
