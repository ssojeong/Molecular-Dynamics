import torch
import os
import numpy as np
import sys
import yaml
# from torchviz import make_dot

from ML.trainer.trainer import trainer
from utils import utils, check_param_dict
from utils.system_logs import system_logs
from utils.mydevice import mydevice
from data_loader.data_loader import data_loader
from data_loader.data_loader import my_data

def main():

    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    torch.set_default_dtype(torch.float64)
    #torch.autograd.set_detect_anomaly(True)

    torch.manual_seed(34952)
    np.random.seed(34952)

    net_type = args['net_type']
    single_parnet_type = args['single_parnet_type']
    multi_parnet_type = args['multi_parnet_type']
    readout_net_type = args['readout_net_type']
    trans_layer = args['trans_layer']
    gnn_layer = args['gnn_layer']
    nnode = args['nnode']
    tau_long = args['tau_long']
    saved_pair_steps = args['saved_pair_steps']
    window_sliding = args['window_sliding']
    batch_size = args['batch_size']
    ngrid = args['ngrid']
    b = args['b']
    a = args['a']
    nitr = args['nitr']
    loss_weights = args['loss_weights']
    ew = args['ew']
    repw = args['repw']
    poly_deg = args['poly_deg']
    maxlr = args['maxlr']
    region = args['region']
    dpt_train = args['dpt_train']
    dpt_valid = args['dpt_valid']
    start_epoch = args['start_epoch']
    loadfile = args['loadfile']

    traindict = {"loadfile"     : loadfile,  # to load previously trained model
                 "net_nnodes"   : nnode,   # number of nodes in neural nets
                 "pw4mb_nnodes" : 128,   # number of nodes in neural nets
                 "pw_output_dim" : 3, # 20250803: change from 2D to 3D, psi
                 "init_weights"   : 'relu', #relu
                 "optimizer" : 'Adam',
                 "single_particle_net_type" : single_parnet_type,         
                 "multi_particle_net_type"  : multi_parnet_type,        
                 "readout_step_net_type"    : readout_net_type,       
                 "n_encoder_layers" : trans_layer,
                 "n_gnn_layers" : gnn_layer,
                 "edge_attention" : True,
                 "d_model"      : 256,
                 "nhead"        : 8,
                 "net_dropout"  : 0.0,    # 1: all zero ; 0 : not dropout ; 0.9 : 90% zero
                 "grad_clip"    : 0.5,    # clamp the gradient for neural net parameters
                 "tau_traj_len" : 8 * tau_long,  # n evaluations in integrator
                 "tau_long"     : tau_long,
                 "saved_pair_steps" : saved_pair_steps, # 20250809 in the saved time points, pair every steps with ai model time step size
                 "loss_weights"  : loss_weights,
                 "window_sliding" : window_sliding,  # number of times to do integration before cal the loss
                 "ngrids"       : ngrid,   # 6*len(b_list)
                 "b_list"       : b,       # grid lattice constant for multibody interactions
                 "a_list"       : a,       #[np.pi/8], #
                 "maxlr"        : maxlr,   # starting learning rate # HK
                 "tau_init"     : 1,       # starting learning rate
                 }

    window_sliding_list = [1,2,3,4,5,7,8,9,10,12,14,16]

    if traindict["window_sliding"] not in window_sliding_list:
        print('window_sliding is not valid, need ',window_sliding_list)
        quit()

    lossdict = { "polynomial_degree" : poly_deg, # 4
                 "rthrsh"            : 0.7,
                 "e_weight"          : ew,
                 "reg_weight"        : repw}

    data = {"train_file": '../data_sets/gen_by_MD/3d/n32lt0.1stpstraj180_l_dpt100.pt', # 20250809
            "valid_file": '../data_sets/gen_by_MD/3d/n32lt0.1stpstraj180_l_dpt10.pt',
           "test_file": '../data_sets/gen_by_MD/3d/n32lt0.1stpstraj180_l_dpt10.pt',
            "train_pts" : dpt_train,
            "vald_pts"  : dpt_valid,
            "test_pts"  : dpt_valid,
             "batch_size": batch_size,
             "window_sliding"   : traindict["window_sliding"] }

    maindict = { "start_epoch"     : start_epoch,
                 "end_epoch"       : 10000,
                  "save_dir"        : './results/traj_len08ws0{}tau{}ngrid{}{}_dpt{}'.format(window_sliding,traindict["tau_long"],ngrid,net_type,dpt_train),
                 "tau_short"       : 1e-4,
                 "nitr"            : nitr, # for check md trajectories # 1000 for ai tau 0.1; 100 for ai tau 0.01
                 "append_strike"   : nitr, # for check md trajectories e.g. based on ground truth tau 0.0001, 100 for tau=0.01, 1000 for tau=0.1
                 "ckpt_interval"   : 10, # for check pointing
                 "val_interval"    : 1, # no use of valid for now
                 "verb"            : 1  } # peroid for printing out losses


    utils.print_dict('trainer', traindict)
    utils.print_dict('loss', lossdict)
    utils.print_dict('data', data)
    utils.print_dict('main', maindict)

    print('begin ------- check param dict -------- ',flush=True)
    check_param_dict.check_maindict(traindict)
    check_param_dict.check_datadict(data)
    check_param_dict.check_traindict(maindict, traindict["tau_long"])
    print('end   ------- check param dict -------- ')

    data_set = my_data(data["train_file"],data["valid_file"],data["test_file"],
                       traindict["tau_long"],traindict["window_sliding"],traindict["saved_pair_steps"],
                       traindict["tau_traj_len"],data["train_pts"],data["vald_pts"],data["test_pts"])
    loader = data_loader(data_set, data["batch_size"])  # 20250908

    #utils.check_data(loader,data_set,traindict["tau_traj_len"],
    #           traindict["tau_long"],maindict["tau_short"],
    #           maindict["nitr"],maindict["append_strike"])

    train = trainer(traindict,lossdict)

    train.load_models()

    print('begin ------- initial learning configurations -------- ')
    train.verbose(0,'init_config')
    print('end  ------- initial learning configurations -------- ')

    for e in range(maindict["start_epoch"], maindict["end_epoch"]):

        cntr = 0
        for qpl_input,qpl_label in loader.train_loader:

            mydevice.load(qpl_input)
            q_traj,p_traj,q_label,p_label,l_init = utils.pack_data(qpl_input, qpl_label)
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
                q_traj,p_traj,q_label,p_label,l_init = utils.pack_data(qpl_input, qpl_label)
                train.eval(q_traj,p_traj,q_label,p_label,l_init)
            train.verbose(e+1,'eval')

    system_logs.print_end_logs()

if __name__=='__main__':
    yaml_config_path = 'main_config.yaml'
    with open(yaml_config_path, 'r') as f:
        args = yaml.load(f, Loader=yaml.Loader)
    main()
