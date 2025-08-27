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
    # torch.autograd.set_detect_anomaly(True)

    torch.manual_seed(34952)
    np.random.seed(34952)

    model_id = args['model_id']
    gap = args['gap']
    tau_long = gap * 0.002
    window_sliding = args['window_sliding']
    ngrid = args['ngrid']
    b = args['b']
    a = args['a']
    nitr = args['nitr']
    ew = args['ew']
    repw = args['repw']
    poly_deg = args['poly_deg']
    maxlr = args['maxlr']
    nnodes = args['nnodes']
    d_model = args['d_model']
    model_name = f"gap{gap}_b{b[0]}_n{'-'.join([str(i) for i in nnodes])}_d{d_model}"
    # ==========================

    traindict = {"net_nnodes"   : args['nnodes'],       # number of nodes in neural nets
                 "pw4mb_nnodes" : 128,                  # number of nodes in neural nets
                 "pw_output_dim": 3,                    # 20250803: change from 2D to 3D, psi
                 "optimizer"    : 'Adam',
                 "single_particle_net_type": args['single_parnet_type'],
                 "multi_particle_net_type" : args['multi_parnet_type'],
                 "readout_step_net_type"   : args['readout_net_type'],
                 "n_encoder_layers" : args['trans_layer'],
                 "n_gnn_layers"     : args['gnn_layer'],
                 "edge_attention"   : True,
                 "d_model"      : d_model,
                 "nhead"        : 8,
                 "net_dropout"  : 0.0,    # 1: all zero ; 0 : not dropout ; 0.9 : 90% zero
                 "grad_clip"    : 0.5,    # clamp the gradient for neural net parameters
                 "tau_traj_len" : 8 * tau_long,  # n evaluations in integrator
                 "tau_long"     : tau_long,
                 "loss_weights"  : args['loss_weights'][-window_sliding:],
                 "window_sliding": window_sliding,  # number of times to do integration before cal the loss
                 "ngrids"       : ngrid,   # 6*len(b_list)
                 "b_list"       : b,       # grid lattice constant for multibody interactions
                 "a_list"       : a,       # [np.pi/8]
                 "maxlr"        : maxlr,   # starting learning rate # HK
                 "tau_init"     : 1,       # starting learning rate
                 }

    lossdict = {"polynomial_degree": poly_deg,
                "rthrsh"           : 0.7,
                "e_weight"         : ew,
                "reg_weight"       : repw}

    data = {"train_file": f'../../Data/LLUF/300k_gap{gap}_train.pt',
            "valid_file": f'../../Data/LLUF/300k_gap{gap}_valid.pt',
            "test_file" : f'../../Data/LLUF/300k_gap{gap}_valid.pt',
            "train_pts" : args['dpt_train'],
            "valid_pts" : args['dpt_valid'],
            "test_pts"  : 200,
            "batch_size": args['batch_size'],
            "window_sliding": window_sliding}

    maindict = { "start_epoch"     : args['start_epoch'],
                 "end_epoch"       : args['end_epoch'],
                 "save_dir"        : f'../../SavedModel/LLUF/{model_name}',
                 "tau_short"       : 1e-4,
                 "nitr"            : nitr,  # for check md trajectories
                 "append_strike"   : nitr,  # for check md trajectories
                 "ckpt_interval"   : 1,     # for check pointing
                 "val_interval"    : 1,     # no use of valid for now
                 "verb"            : 1  }   # period for printing out losses

    os.makedirs(maindict['save_dir'], exist_ok=True)

    if maindict['start_epoch'] == 0:
        traindict['loadfile'] = None
    else:
        traindict['loadfile'] = f"{maindict['save_dir']}/{model_id}_{maindict['start_epoch']:06d}.pth"

    utils.print_dict('trainer', traindict)
    utils.print_dict('loss', lossdict)
    utils.print_dict('data', data)
    utils.print_dict('main', maindict)

    print('begin ------- check param dict -------- ', flush=True)
    check_param_dict.check_traindict(traindict)
    check_param_dict.check_datadict(data)
    # check_param_dict.check_maindict(maindict, traindict["tau_long"])
    print('end   ------- check param dict -------- ')

    data_set = my_data(data["train_file"], data["valid_file"], data["test_file"],
                       traindict["tau_long"], traindict["window_sliding"], traindict["tau_traj_len"],
                       data["train_pts"], data["valid_pts"], data["test_pts"])
    loader = data_loader(data_set, data["batch_size"])

    train = trainer(traindict, lossdict)

    train.load_models()

    print('begin ------- initial learning configurations -------- ')
    train.verbose(0, 'init_config')
    print('end  ------- initial learning configurations -------- ')

    for e in range(maindict["start_epoch"], maindict["end_epoch"]):

        cntr = 0
        for qpl_input, qpl_label in loader.train_loader:

            mydevice.load(qpl_input)
            q_traj, p_traj, q_label, p_label, l_init = utils.pack_data(qpl_input, qpl_label)
            # q_traj, p_ttaj [traj, nsamples, nparticles, dim]
            # q_label, p_label, l_init [nsamples, nparticles, dim]

            train.one_step(q_traj, p_traj, q_label, p_label, l_init)
            cntr += 1
            if cntr % 10 == 0:
                print('.', end='', flush=True)

        print(cntr, 'batches \n')

        if e % maindict["verb"] == 0:
            train.verbose(e+1, 'train')
            system_logs.record_memory_usage(e+1)
            print('time use for ', maindict["verb"], 'epoches is: ', end='')
            system_logs.record_time_usage(e+1)

        if e % maindict["ckpt_interval"] == 0:
            filename = f"./{maindict['save_dir']}/{model_id}_{e+1:06d}.pth"
            print('saving file to ', filename)
            train.checkpoint(filename)

        if e % maindict["val_interval"] == 0:
            train.loss_obj.clear()
            for qpl_input, qpl_label in loader.val_loader:
                q_traj, p_traj, q_label, p_label, l_init = utils.pack_data(qpl_input, qpl_label)
                train.eval(q_traj, p_traj, q_label, p_label, l_init)
            train.verbose(e+1, 'eval')

    system_logs.print_end_logs()


if __name__ == '__main__':
    yaml_config_path = 'config.yaml'
    with open(yaml_config_path, 'r') as f:
        args = yaml.load(f, Loader=yaml.Loader)
    main()
