import torch
import os
import numpy as np
import sys

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

    argv = sys.argv
    if len(argv) != 25:
        print('usage <programe> <net type> <single net type> \
              <multi net type> <readout net type> <trans layer> <gnn layer> \
              <nnode> <tau_long> <window sliding> <batchsize> <ngrid> <b> <a> \
              <nitr> <loss weight> <ew> <repw> <poly_deg> <lr> <region> <dpt_train> <dpt_valid> \
              <start epoch> <filename>' )
        quit()

    net_type = argv[1]
    single_parnet_type = argv[2]
    multi_parnet_type = argv[3]
    readout_net_type = argv[4]
    trans_layer = int(argv[5])
    gnn_layer = int(argv[6])
    nnode = int(argv[7])
    tau_long = float(argv[8])
    window_sliding = int(argv[9])
    batch_size = int(argv[10])
    ngrid = int(argv[11])
    b = argv[12].split(',')
    a = argv[13].split(',')
    nitr = int(argv[14])
    loss_weights = argv[15].split(',')
    ew = int(argv[16])
    repw = int(argv[17])
    poly_deg = int(argv[18])
    maxlr = float(argv[19])
    region = argv[20]
    dpt_train = int(argv[21])
    dpt_valid = int(argv[22])
    start_epoch = int(argv[23])
    loadfile = argv[24]


    for i in range(len(loss_weights)):
        if isinstance(loss_weights[i], float):
            loss_weights[i] = float(loss_weights[i])
        else:
            loss_weights[i] = eval(loss_weights[i])

    for i in range(len(b)):
        if isinstance(b[i], float):
            b[i] = float(b[i])
        else:
            b[i] = eval(b[i],{'np':np})

    for i in range(len(a)):

        if isinstance(a[i], float):
            a[i] = float(a[i])
        else:
            a[i] = eval(a[i],{'np':np})


    if loadfile.strip() == "None":
       loadfile = None
    else:
       loadfile = loadfile.strip()

    traindict = {"loadfile"     : loadfile,  # to load previously trained model
                 "net_nnodes"   : nnode,   # number of nodes in neural nets
                 "pw4mb_nnodes" : 128,   # number of nodes in neural nets
                 "pw_output_dim" : 3, # 20250803: change from 2D to 3D, psi
                 "init_weights"   : 'tanh', #relu
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

    data = {"train_file": '../data_sets/gen_by_MD/3d/n32lt{}stpstraj18_l_dpt100.pt'.format(tau_long),
            "valid_file": '../data_sets/gen_by_MD/3d/n32lt{}stpstraj18_l_dpt10.pt'.format(tau_long),
            "test_file": '../data_sets/gen_by_MD/3d/n32lt{}stpstraj18_l_dpt10.pt'.format(tau_long),
            "train_pts" : 10,
            "vald_pts"  : 10,
            "test_pts"  : 10,
             "batch_size": batch_size,
             "window_sliding"   : traindict["window_sliding"] }

    maindict = { "start_epoch"     : start_epoch,
                 "end_epoch"       : 10,
                 # "save_dir"        : './results/traj_len08ws0{}tau{}ngrid{}{}_dpt{}'.format(window_sliding,traindict["tau_long"],ngrid,net_type,dpt_train),
                 "save_dir"        : './results',
                 "tau_short"       : 1e-4,
                 "nitr"            : nitr, # for check md trajectories
                 "append_strike"   : nitr, # for check md trajectories
                 "ckpt_interval"   : 1, # for check pointing
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
                       traindict["tau_long"],traindict["window_sliding"],traindict["tau_traj_len"],
                       data["train_pts"],data["vald_pts"],data["test_pts"])
    loader = data_loader(data_set, data["batch_size"])

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

# python maintrain09.py api0lw8421ew1repw10poly1lowrho transformer_type gnn_identity mlp_type
# 2 2 128 0.1 8 10 6 0.2 0 1000 0,1/8,0,1/4,0,1/2,0,1 1 10 1 1e-5 g 540000 60000 0 None
if __name__=='__main__':
    main()


