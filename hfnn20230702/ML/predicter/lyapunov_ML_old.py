import sys 
sys.path.append('../../')
from utils.system_logs              import system_logs
from utils.mydevice                 import mydevice
from utils                       import utils
from utils.pbc                      import single_particle_dq_pbc
from hamiltonian.lennard_jones2d    import lennard_jones2d
from MD.velocity_verlet_MD          import velocity_verlet_MD
from ML.trainer.trainer          import trainer
from ML.predicter.predicter      import predicter
import torch
import itertools

def pack_data(qpl_list):

    q_traj = qpl_list[:,0,0,:,:].clone().detach()
    p_traj = qpl_list[:,1,0,:,:].clone().detach()
    l_init = qpl_list[:,2,0,:,:].clone().detach()

    q_traj = mydevice.load(q_traj)
    p_traj = mydevice.load(p_traj)
    l_init = mydevice.load(l_init)

    return q_traj,p_traj,l_init

def norm_dq(q_list, q_eps_list, l_list):

    nsamples, nparticle, _ = q_list.shape
    # shape = [nsamples, nparticles, DIM]

    dq = single_particle_dq_pbc(q_list, q_eps_list, l_list)
    # shape = [nsamples, nparticles, DIM]

    dq_sqrt = torch.sqrt(torch.sum(dq * dq, dim=2))
    # shape = [nsamples, nparticles]

    mean_dq_sqrt = torch.sum(dq_sqrt, dim=1) / nparticle
    # shape = [nsamples]

    return mean_dq_sqrt

def lyapunov_exp(q_list, q_eps_list, l_list):
    dq_list = norm_dq(q_list, q_eps_list, l_list)
    # shape = [nsamples]
    return dq_list

if __name__ == '__main__':

    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    argv = sys.argv
    if len(argv) != 8:
        print('usage <programe> <npar> <rho> <temp> <level> <tau max> <saved model> <name>' )
        quit()

    param1 = argv[1]
    param2 = argv[2]
    param3 = argv[3]
    param4 = argv[4]
    param5 = float(argv[5])
    param6 = argv[6]
    param7 = argv[7]

    states = { "npar" : param1,
               "rho"  : param2,
               "T"    : param3,
               "level": param4,
               "t_max" : param5,
               "num_saved": param6,
               "name" : param7
              }

    npar = states["npar"]
    rho = states["rho"]
    T = states["T"]
    level = int(states["level"])
    t_max =  states["t_max"]
    num_saved = str(states["num_saved"])
    name = states["name"]

    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(34952)

    traindict = {"loadfile": '../../results20230409/traj_len08nchain0{}tau0.1d256l2ew01repw10_dpt1800000/mbpw000{}.pth'.format(level,num_saved),  # to load previously trained model
                  # "loadfile": "results20230314/traj_len08nchain01tau0.1d256l2ew01repw10lowrho_dpt1152000/mbpw000159.pth",
                  "pwnet_nnodes": 128,  # number of nodes in neural nets for force
                  "mbnet_nnodes": 128,  # number of nodes in neural nets for momentum
                  "pw4mb_nnodes": 128,  # number of nodes in neural nets for momentum
                  "h_mode": 'relu',
                  # "h_mode"       : 'tanh',
                  # "mb_type"       : "mlp_type",
                  "mb_type": "transformer_type",  # ---LW
                  "d_model": 256,  # ---LW
                  "nhead": 8,  # ---LW
                  "n_encoder_layers": 2,  # ---LW
                  "mbnet_dropout": 0.5,  # 1: all zero ; 0 : not dropout ; 0.9 : 90% zero
                  "grad_clip": 0.5,  # clamp the gradient for neural net parameters
                  "tau_traj_len": 8 * 0.1,  # n evaluations in integrator
                  "tau_long": 0.1,
                  "n_chain": level,  # number of times to do integration before cal the loss
                  "ngrids": 6,  # for multibody interactions
                  "b": 0.2,  # grid lattice constant for multibody interactions
                  # "maxlr"        : 1e-5,  # starting learning rate # HK
                  # "minlr"        : 1e-7,  # starting learning rate # HK
                  "maxlr": 1e-5,  # starting learning rate # HK
                  "minlr": 1e-5,  # starting learning rate # HK
                  "tau_lr": 1e-2,  # starting learning rate
                  "tau_init": 1,  # starting learning rate
                  # "sch_step"     : 10,    # scheduler step
                  # "sch_decay"    : 0.9,   # 200 steps and reduce lr to 10%
                  "sch_step": 1,  # scheduler step
                  "sch_decay": 1,  # 200 steps and reduce lr to 10%
                  "reset_lr": False}
                 # "reset_lr"     : True}

    lossdict = { "polynomial_degree" : 2,
                 "rthrsh"       : 0.7,
                 "e_weight"    : 1,
                 "reg_weight"    : 10}

    data = {"filename": "../../../data_sets/gen_by_MD/noML-metric-st1e-4every0.1t100/n{}rho{}T{}/".format(npar,rho,T)
                                       + 'n{}rho{}T{}.pt'.format(npar,rho,T),
	        # "dt_init_config_filename": "../../../data_sets/gen_by_MD/noML-metric-lt0.01every0.1t0.7t100/n{}rho{}T{}/".format(npar,rho,T)
            #                             + 'n{}rho{}T{}_'.format(npar, rho, T),
            "saved_filename" : "../../../data_sets/gen_by_ML/lt0.1dpt1800000/n{}rho{}T{}/".format(npar,rho,T) + "{}".format(states["name"])
            }

    maindict = {"dt"  : 1e-4,
                "eps" : 2e-4,
	            "dt_samples" : 1 ,# no. of eps samples
                "everysave" : 0.1
                }

    saved_filename = data["saved_filename"]
    # dt_init_config_filename = data["dt_init_config_filename"]
    tau_traj_len = traindict["tau_traj_len"]
    traj_len_idx = round(tau_traj_len / traindict["tau_long"])
    label_idx = int((traj_len_idx - 1) + level)
    dt = maindict['dt']
    eps = maindict['eps']
    everysave = maindict["everysave"]
    nsteps = round((t_max - ((traj_len_idx - 1) * traindict["tau_long"])) / traindict["tau_long"])
    t_thrsh = nsteps # thrsh not over t incremented until eps

    print('nsteps ', nsteps, 'label idx', label_idx, 't thrsh ', t_thrsh)
    assert (nsteps >= t_thrsh), '....incompatible nsteps and t thrsh'

    lj_obj = lennard_jones2d()
    mdvv = velocity_verlet_MD(lj_obj)

    train = trainer(traindict,lossdict)
    train.load_models()

    predict = predicter(train.mbpw_obj)

    for j in range(0, maindict['dt_samples']):

        print('sample for dt traj ====' , j )
        print('guess t max ', t_max )

        thrsh = [0]
        R_sample_append = []
        dq_append = []
        avg_dq_append = []

        print('t thrsh ======', thrsh, flush=True)

        print('start t =======' )

        qpl_list = torch.load(data["filename"], map_location=map_location)
        qpl_traj = qpl_list['qpl_trajectory'][:10]
        qpl_input = qpl_traj[:,:,0:traj_len_idx,:,:]
        #shape [nsamples, 3, traj, nparticles, dim]
        print('load data init file : ',  qpl_input.shape)
        qpl_label = qpl_traj[:, :,label_idx, :, :]

        q_traj, p_traj, q_label, p_label, l_init = utils.pack_data(qpl_input, qpl_label)
        q_input_list, p_input_list, q_cur_, p_cur_ = predict.prepare_input_list(q_traj, p_traj, l_init)
        qpl_in = torch.unsqueeze(torch.stack((q_cur_, p_cur_, l_init), dim=1), dim=2)

        qpl_dt_in = qpl_in + mydevice.load(torch.FloatTensor(qpl_in.shape).uniform_(-maindict['dt'], maindict['dt']))
        # shape [nsamples,3,1,nparticles,dim]

        q_init, p_init, l_list = pack_data(qpl_in)
        q_dt_init, p_dt_init, _ = pack_data(qpl_dt_in)
        print('q init shape', q_init.shape, 'q dt init shape', q_dt_init.shape)

        prepare_q_input_netid = 0

        q_input_cur = predict.mbpw_obj.prepare_q_input(prepare_q_input_netid,q_dt_init, p_dt_init, l_list)
        p_input_cur = predict.mbpw_obj.prepare_p_input(q_dt_init, p_dt_init, l_list)

        q_dt_input_list = q_input_list.copy()
        q_dt_input_list.pop(-1)
        q_dt_input_list.append(q_input_cur)

        p_dt_input_list = p_input_list.copy()
        p_dt_input_list.pop(-1)
        p_dt_input_list.append(p_input_cur)

        dq_init = lyapunov_exp(q_init, q_dt_init, l_list)
        # shape = [nsamples] 1 is initial

        avg_dq_init =  torch.sum(dq_init, dim=0) /  dq_init.shape[0]
        print('sample avg init dq list', avg_dq_init)

        l1_sample = dq_init  # shape [nsamples]
        l1_sample_avg = avg_dq_init  # shape []

        dq_append.append(dq_init) # shape [nsamples]
        avg_dq_append.append(l1_sample_avg.item()) # shape []

        q_cur = q_init
        p_cur = p_init
        q_dt_cur = q_dt_init
        p_dt_cur = p_dt_init

        for k in range(t_thrsh): # thrsh not over t incremented until eps

            taccum = itertools.accumulate(thrsh)
            t_accum = list(taccum)
            print('t accum ======', t_accum)

            print('increment t until eps =======', k+1, flush=True)
            #print('before iter', q_cur[0])
            q_next_list1, p_next_list1, q_list1, p_list1, l_list1 = predict.mlvv.one_step(q_input_list, p_input_list,q_cur, p_cur,l_list)

            q_next_list2, p_next_list2, q_list2, p_list2, l_list2 = predict.mlvv.one_step(q_dt_input_list, p_dt_input_list, q_dt_cur, p_dt_cur, l_list)
            #print(qpl_list1.shape, qpl_list2.shape)

            dq_list = lyapunov_exp(q_list1, q_list2, l_list1) # along time
            # shape [nsmaples]

            #print('GPU memory % allocated:', round(torch.cuda.memory_allocated(0)/1024**3,2) ,'GB', '\n')
            avg_dq_list =  torch.sum(dq_list, dim=0) /  dq_list.shape[0]
            print('sample avg dq list', avg_dq_list)
            # shape []

            q_cur = q_list1
            p_cur = p_list1
            l_list = l_list1
            q_dt_cur = q_list2
            p_dt_cur = p_list2
            q_input_list = q_next_list1
            p_input_list = p_next_list1
            q_dt_input_list = q_next_list2
            p_dt_input_list = p_next_list2

            if avg_dq_list < eps: # 4e-2 4e-3
                print('L < eps .....') #, avg_dq_list)

                if k+1 == t_thrsh:

                    print('Reach the end of t thrsh .....') #, avg_dq_list)
                    l2_sample = dq_list  # shape [nsamples]]
                    l2_sample_avg = avg_dq_list  # shape []

                    R_sample = l2_sample / l1_sample # shape [nsamples]
                    R_avg = l2_sample_avg / l1_sample_avg # shape []

                    thrsh.append(k+1)
                    R_sample_append.append(R_sample)
                    avg_dq_append.append(l2_sample_avg.item())

                else:
                    continue

            else:

                print('L > eps .....', avg_dq_list.item(), 't =', k+1)
                l2_sample = dq_list  # shape [nsamples]
                l2_sample_avg = avg_dq_list  # shape []

                R_sample = l2_sample / l1_sample # shape [nsamples]
                R_avg = l2_sample_avg / l1_sample_avg # shape []

                thrsh.append(k+1)
                R_sample_append.append(R_sample)
                avg_dq_append.append(l2_sample_avg.item())
                #print('increment t ===== ', k+1, l1, LogR)

                print(k+1, R_sample_append)
                data = {'t_accum': k+1, 'R_sample_append': R_sample_append}
                torch.save(data, saved_filename)
                print('save file dir', saved_filename)

                quit()
                break

system_logs.print_end_logs()

