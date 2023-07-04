import torch
import os
import numpy as np

# from torchviz import make_dot

from ML.trainer.trainer          import trainer
from utils                       import check_param_dict
from utils                       import utils
from utils.system_logs           import system_logs
from utils.mydevice              import mydevice
from data_loader.data_loader     import data_loader
from data_loader.data_loader     import my_data

def main():

    #os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    torch.set_default_dtype(torch.float64)
    #torch.autograd.set_detect_anomaly(True)

    #print('checking pbc method ... wait a minute ')
    #check_pbc()
    torch.manual_seed(34952)
    np.random.seed(34952)

    traindict = {"loadfile"     : None,  # to load previously trained model
                 #"loadfile": "results20230409/traj_len08nchain08tau0.1d256l2ew01repw10_dpt1800000/mbpw000081.pth",
                 "pw_on_off"    : "off",
                 "pwnet_nnodes" : 128,   # number of nodes in neural nets for force
                 "mbnet_nnodes" : 128,   # number of nodes in neural nets for momentum
                 "pw4mb_nnodes" : 128,   # number of nodes in neural nets for momentum
                 "h_mode"       : 'relu',
                 #"h_mode"       : 'tanh',
                 # "mb_type"     : "mlp_type",
                 "mb_type"      : "transformer_type",         # ---LW
                 "d_model"      : 256,                      # ---LW
                 "nhead"        : 8,                         # ---LW
                 "n_encoder_layers"  : 2,                         # ---LW
                 "mbnet_dropout": 0.5,   # 1: all zero ; 0 : not dropout ; 0.9 : 90% zero
                 "grad_clip"    : 0.5,  # clamp the gradient for neural net parameters
                 "tau_traj_len" : 8*0.1,  # n evaluations in integrator
                 "tau_long"     : 0.1,
                 "n_chain"      : 8,     # number of times to do integration before cal the loss
                 "ngrids"       : 6,     # for multibody interactions
                 "b"            : 0.2,   # grid lattice constant for multibody interactions
                 #"maxlr"        : 1e-5,  # starting learning rate # HK
                 #"minlr"        : 1e-7,  # starting learning rate # HK
                 "maxlr"        : 1e-5,  # starting learning rate # HK
                 "minlr"        : 1e-5,  # starting learning rate # HK
                 "tau_lr"       : 1e-2,  # starting learning rate
                 "tau_init"     : 1,  # starting learning rate
                 #"sch_step"     : 10,    # scheduler step
                 #"sch_decay"    : 0.9,   # 200 steps and reduce lr to 10%
                 "sch_step"     : 1,    # scheduler step
                 "sch_decay"    : 1,   # 200 steps and reduce lr to 10%
                 "reset_lr"     : False}
                 #"reset_lr"     : True}

    n_chain_list = [1,2,3,4,5,7,8,9,10,12,14,16]

    if traindict["n_chain"] not in n_chain_list:
        print('n_chain is not valid, need ',n_chain_list)
        quit()

    lossdict = { "polynomial_degree" : 4,
                 "rthrsh"       : 0.7,
                 "e_weight"    : 1,
                 "reg_weight"    : 10}

    data = { "train_file": '../data_sets/n16lt0.1stpstraj24_s1000.pt',
             "valid_file": '../data_sets/n16lt0.1stpstraj24_s100.pt',
             "test_file" : '../data_sets/n16lt0.1stpstraj24_s100.pt',
    #data = { "train_file": '../data_sets/n16lt0.2stpstraj24_s1800000.pt',
    #         "valid_file": '../data_sets/n16lt0.2stpstraj24_s200000.pt',
    #         "test_file" : '../data_sets/n16lt0.2stpstraj24_s200000.pt',
             "train_pts" : 100,
             "vald_pts"  : 10,
             "test_pts"  : 10,
             "batch_size": 10,
             "n_chain"   : traindict["n_chain"] }

    maindict = { "start_epoch"     : 0,
                 "end_epoch"       : 4,
                 "grad_clip_reset_epoch" : 100, # reset clip_grad = value after XX epoch
                 "grad_clip_reset_value" : 1,
                 #"save_dir": './results20230224/test',
                 "save_dir"        : './results20230410/traj_len08nchain0{}tau{}d{}l{}ew01repw10_dpt1800000'.format(traindict["n_chain"],traindict["tau_long"],traindict["d_model"],traindict["n_encoder_layers"]),
                 "nitr"            : 1000, # for check md trajectories
                 "tau_short"       : 1e-4,
                 "append_strike"   : 1000, # for check md trajectories
                 "ckpt_interval"   : 2, # for check pointing
                 "val_interval"    : 1, # no use of valid for now
                 "verb"            : 1  } # peroid for printing out losses

    #if not os.path.exists(maindict["save_dir"]):
    #   os.makedirs(maindict["save_dir"])

    utils.print_dict('trainer',traindict)
    utils.print_dict('loss',lossdict)
    utils.print_dict('data',data)
    utils.print_dict('main',maindict)

    print('begin ------- check param dict -------- ',flush=True)
    check_param_dict.check_maindict(traindict)
    check_param_dict.check_datadict(data)
    check_param_dict.check_traindict(maindict,traindict["tau_long"])
    print('end   ------- check param dict -------- ')

    data_set = my_data(data["train_file"],data["valid_file"],data["test_file"],
                       traindict["tau_long"],traindict["n_chain"],traindict["tau_traj_len"],
                       data["train_pts"],data["vald_pts"],data["test_pts"])
    loader = data_loader(data_set, data["batch_size"])

    #utils.check_data(loader,data_set,traindict["tau_traj_len"],
    #           traindict["tau_long"],maindict["tau_short"],
    #           maindict["nitr"],maindict["append_strike"])

    train = trainer(traindict,lossdict)

    train.load_models()
    if traindict["reset_lr"] is True: train.reset_opt_lr(traindict["maxlr"]) # minlr -> maxlr

    print('begin ------- initial learning configurations -------- ')
    train.verbose(0,'init_config')
    print('end  ------- initial learning configurations -------- ')

    for e in range(maindict["start_epoch"], maindict["end_epoch"]):

        cntr = 0
        for qpl_input,qpl_label in loader.train_loader:

            mydevice.load(qpl_input)
            q_traj,p_traj,q_label,p_label,l_init = utils.pack_data(qpl_input,qpl_label)
            # q_traj,p_ttaj [traj,nsamples,nparticles,dim]
            # q_label,p_label,l_init [nsamples,nparticles,dim]

            train.one_step(q_traj,p_traj,q_label,p_label,l_init)
            cntr += 1
            if cntr%10==0: print('.',end='',flush=True)

        print(cntr,'batches \n')

        if e%maindict["verb"]==0: 
            train.verbose(e+1,'train')
            system_logs.record_memory_usage(e+1)
            print('time use for ',maindict["verb"],'epoches is: ',end='')
            system_logs.record_time_usage(e+1)

        if e%maindict["ckpt_interval"]==0: 
            filename = './{}/mbpw{:06d}.pth'.format(maindict["save_dir"],e+1)
            print('saving file to ',filename)
            train.checkpoint(filename)

        if e%maindict["val_interval"]==0: 
            train.loss_obj.clear()
            for qpl_input,qpl_label in loader.val_loader:
                q_traj,p_traj,q_label,p_label,l_init = utils.pack_data(qpl_input,qpl_label)
                train.eval(q_traj,p_traj,q_label,p_label,l_init)
            train.verbose(e+1,'eval')

        if traindict["sch_step"] > 0:
            train.scheduler_step()

        if e>maindict["grad_clip_reset_epoch"] and e%10==0:
            train.try_reset_grad_clip(maindict["grad_clip_reset_value"])

    system_logs.print_end_logs()

if __name__=='__main__':
    main()


