import torch
import time
import sys
#from torchviz import make_dot
from ML.trainer.trainer          import trainer
from ML.predicter.predicter      import predicter
from utils                       import check_param_dict
from utils                       import utils
from utils.system_logs           import system_logs
from utils.mydevice              import mydevice
from utils.pbc                   import pbc
from data_loader.check_load_data import check_load_data


def del_q_adjust(q_list, q_label, l_list):

    dq = q_list - q_label
    # shape [nsamples, nparticle, DIM]
    dq = pbc(dq, l_list)
    return dq

def q_RMSE(q_list, q_label, l_list):

    nsamples, nparticle, DIM = q_label.shape
    dq = del_q_adjust(q_list,q_label, l_list) # shape is [nsamples, nparticle, DIM]
    d2 = torch.sqrt(torch.sum(dq * dq, dim=2)) # shape is [nsamples, nparticle]
    dq_sqrt = torch.sum(d2,dim=1) / nparticle # shape [nsamples]
    return dq_sqrt

def p_RMSE(p_list, p_label):

    nparticles = p_list.shape[1]
    dp = p_list - p_label # shape [nsamples,nparticles,dim]
    dp2 = torch.sqrt(torch.sum(dp*dp, dim = 2))  # shape [nsamples,nparticles]
    dp_sqrt = torch.sum(dp2, dim = 1) / nparticles # shape [nsamples]
    return dp_sqrt  


if __name__=='__main__':

    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    torch.set_default_dtype(torch.float64)

    torch.manual_seed(34952)

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
               "tau_max": param5,
               "num_saved" : param6,
               "name" : param7   #"pred_len08C4d256l2mbpw231_tau"
              }

    npar = states["npar"]
    rho = states["rho"]
    T = states["T"]
    level = int(states["level"])
    tau_max =  states["tau_max"]
    num_saved = str(states["num_saved"])
    name = states["name"]

    traindict = {"loadfile": './results20230409/traj_len08nchain0{}tau0.2d128l2ew01repw10_dpt1800000/mbpw000{}.pth'.format(level,num_saved),  # to load previously trained model
                  "pwnet_nnodes": 128,  # number of nodes in neural nets for force
                  "mbnet_nnodes": 128,  # number of nodes in neural nets for momentum
                  "pw4mb_nnodes": 128,  # number of nodes in neural nets for momentum
                  "h_mode": 'relu',
                  # "h_mode"       : 'tanh',
                  # "mb_type"       : "mlp_type",
                  "mb_type": "transformer_type",  # ---LW
                  "d_model": 128,  # ---LW
                  "nhead": 8,  # ---LW
                  "n_encoder_layers": 2,  # ---LW
                  "mbnet_dropout": 0.5,  # 1: all zero ; 0 : not dropout ; 0.9 : 90% zero
                  "grad_clip": 0.5,  # clamp the gradient for neural net parameters
                  "tau_traj_len": 8 * 0.2,  # n evaluations in integrator
                  "tau_long": 0.2,
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

    data = { 
            'test_file' : '../data_sets/gen_by_MD/noML-metric-st1e-4every0.1t100/n{}rho{}T{}'.format(npar,rho,T)
             + '/n{}rho{}T{}.pt'.format(npar,rho,T)}

    maindict = {
                 "Ssample"      : 50,
                 "save_dir"     : '../data_sets/gen_by_ML/lt{}dpt1800000/n{}rho{}T{}/'.format(traindict["tau_long"],npar,rho,T) + "{}".format(states["name"]),
                 "nitr"         : 2000,  # for check md trajectories
                 "tau_short"    : 1e-4,
                 "append_strike": 2000, # for check md trajectories
                 "everytau"     : traindict["tau_long"],
                 "everysave"     : 0.1
               }

    utils.print_dict('data',data)

    print(traindict)
    print('begin ------- check param dict -------- ')
    check_param_dict.check_maindict(traindict)
    check_param_dict.check_traindict(maindict,traindict["tau_long"])
    #check_param_dict.check_testdict(maindict)
    #assert ( traindict["tau_traj_len"] == maindict['tau_max'] ), "not match tau traj len and tau_max"
    print('end   ------- check param dict -------- ')

    qpl_list    = torch.load(data["test_file"])
    qpl_input   = qpl_list['qpl_trajectory']
    print('qpl input', qpl_input.shape)

    tau_traj_len = traindict["tau_traj_len"]
    tau_long = traindict["tau_long"]
    everysave = maindict["everysave"]
    everytau = maindict["everytau"]
    n_chain = traindict["n_chain"]
    traj_len_idx = round(tau_traj_len / everysave )
    one_step_idx = round(everytau / everysave)
    label_idx = int((traj_len_idx- one_step_idx) + n_chain * one_step_idx)
    Ssample = maindict["Ssample"]

    print('one step idx', one_step_idx, 'traj len idx', traj_len_idx, 'label idx', label_idx)

    tau_traj_prep = round(tau_traj_len - tau_long, 4)  # e.g. tau_traj_len=4*2 , tau_traj_prep = 8 - 2

    qpl_init    = qpl_input[:,:,0:traj_len_idx:one_step_idx,:,:]
    # qpl_init [nsamples,3,traj,nparticles,dim]
    print('qpl init', qpl_init.shape)

    qpl_label   = qpl_input[:,:,label_idx,:,:]

    check_load_data = check_load_data(qpl_init, qpl_label)

    q_traj, p_traj, q_label, p_label, l_init = utils.pack_data(qpl_init, qpl_label)

    q_init_traj,p_init_traj,_,_,_ = utils.pack_data(qpl_init,qpl_label)
    # q_traj,p_ttaj [traj,nsamples,nparticles,dim]

    S_ML    = int((tau_max - tau_traj_prep) / tau_long / n_chain) # ML steps
    print('tau traj prep ', tau_traj_prep, 'predict ml step ', S_ML)

    train = trainer(traindict,lossdict)
    train.load_models()
    #mbpw_obj, _ = train.net_builder(traindict)

    predict = predicter(train.mbpw_obj)

    cntr = 0
    qpl_list_sample = []
    for i in range(0,q_init_traj.shape[1],Ssample):
        print('====== sample ', i * Ssample + Ssample, flush=True)
        qpl_batch = []

        q_next_traj = q_init_traj[:,i:i+Ssample]
        p_next_traj = p_init_traj[:,i:i+Ssample]
        l_next = l_init[i:i+Ssample]

        q_input_list,p_input_list,q_cur,p_cur=predict.prepare_input_list(q_next_traj,p_next_traj,l_next)
        qpl_in = torch.unsqueeze(torch.stack((q_cur,p_cur,l_next),dim=1),dim=2)

        start_time = time.time()
        for t in range(S_ML):

            print('====== t=',round(tau_traj_prep + t * n_chain * tau_long,3),' window sliding ', n_chain,
                  't=', round((t+1) * n_chain * tau_long + tau_traj_prep,3), flush=True)
            #print('q',q_next.shape,'p', p_next.shape,'l', l_next.shape)

            q_next_list,p_next_list, qpl_list = predict.eval(q_input_list,p_input_list,q_cur,p_cur,l_next,traindict["n_chain"])
            # qpl_list [[nsamples,3,nparticles,dim], [nsamples,3,nparticles,dim]....,]

            qpl_list_torch = torch.stack(qpl_list, dim=2)  # shape [nsamples,3, traj_len, nparticles,dim]
            qpl_batch.append(qpl_list_torch)

            q_pred = qpl_list_torch[:, 0, -1, :, :]  # shape [nsamples,nparticles,dim]
            p_pred = qpl_list_torch[:, 1, -1, :, :]
            l_list = qpl_list_torch[:, 2, -1, :, :]

            q_cur = q_pred
            p_cur = p_pred
            l_next = l_list

            #if t >= 1: quit()
            # print('t= ',t, 'CPU memory % used:', psutil.virtual_memory()[2], '\n')
            cntr += 1
            if cntr%10==0: print('.', end='', flush=True)

        sec = time.time() - start_time
        #sec = sec / maindict["nitr"]
        #mins, sec = divmod(sec, 60)
        print("{} nitr --- {:.03f} sec ---".format(S_ML,sec))
        print("samples {}, one forward step timing --- {:.03f} sec ---".format(Ssample, sec/S_ML))

        qpl_batch = torch.cat(qpl_batch, dim=2) # shape [nsamples,3, traj_len, nparticles,dim]
        # qpl_batch [nsamples,3,traj,nparticles,dim]
        print(qpl_in.shape,qpl_batch.shape)
        qpl_batch_cat = torch.cat((qpl_in,qpl_batch), dim=2) # stack traj
        qpl_list_sample.append(qpl_batch_cat)
        # if i == 3*step: quit()

    qpl_list2 = torch.cat(qpl_list_sample, dim=0)
    # qpl_list [nsamples,3,traj,nparticles,dim]
    print('saved qpl list shape', qpl_list2.shape)

    torch.save({'qpl_trajectory': qpl_list2, 'tau_short':maindict['tau_short'], 'tau_long' : traindict["tau_long"]}, maindict["save_dir"] + str(traindict['tau_long']) + '.pt')

system_logs.print_end_logs()


