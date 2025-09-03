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


def main(args, model_name):

    _ = mydevice()

    torch.set_default_dtype(torch.float64)

    torch.manual_seed(34952)
    np.random.seed(34952)

    model_id = args.model_id
    gap = args.gap
    tau_long = gap * 0.002
    window_sliding = args.window_sliding
    ngrid = args.ngrid
    b = args.b
    a = args.a
    nitr = args.nitr
    ew = args.ew
    repw = args.repw
    poly_deg = args.poly_deg
    maxlr = args.maxlr
    d_model = args.trans_dim
    # ==========================
    nnodes = [args.pwnet_dim] * args.pwnet_layer
    # ==========================

    traindict = {"net_nnodes"   : nnodes,       # number of nodes in neural nets
                 "pw4mb_nnodes" : 128,                  # number of nodes in neural nets
                 "pw_output_dim": 3,                    # 20250803: change from 2D to 3D, psi
                 "optimizer"    : 'Adam',
                 "single_particle_net_type": args.single_parnet_type,
                 "multi_particle_net_type" : args.multi_parnet_type,
                 "readout_step_net_type"   : args.readout_net_type,
                 "n_encoder_layers" : args.trans_layer,
                 "n_gnn_layers"     : args.gnn_layer,
                 "edge_attention"   : True,
                 "d_model"      : d_model,
                 "nhead"        : 8,
                 "net_dropout"  : 0.0,    # 1: all zero ; 0 : not dropout ; 0.9 : 90% zero
                 "grad_clip"    : 0.5,    # clamp the gradient for neural net parameters
                 "tau_traj_len" : 8 * tau_long,  # n evaluations in integrator
                 "tau_long"     : tau_long,
                 "loss_weights"  : args.loss_weights[-window_sliding:],
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
            "train_pts" : args.dpt_train,
            "valid_pts" : args.dpt_valid,
            "test_pts"  : 1000,
            "batch_size": args.batch_size,
            "window_sliding": window_sliding}

    maindict = {"end_epoch"       : args.end_epoch,
                "save_dir"        : f'../../SavedModel/LLUF/{model_name}',
                "tau_short"       : 1e-4,
                "nitr"            : nitr,  # for check md trajectories
                "append_strike"   : nitr,  # for check md trajectories
                "ckpt_interval"   : 10,     # for check pointing
                "val_interval"    : 1,     # no use of valid for now
                "verb"            : 1  }   # period for printing out losses

    os.makedirs(maindict['save_dir'], exist_ok=True)

    log_file_path = f"./logfile/{model_name}_{model_id}.log"
    _ = system_logs(mydevice, log_file_path)
    system_logs.print_start_logs()

    if args.load_weight.lower() in ('null', 'none'):
        traindict['loadfile'] = None
    elif args.load_weight.lower() == 'best':
        traindict['loadfile'] = f"{maindict['save_dir']}/{model_id}_best.pth"
    else:
        traindict['loadfile'] = f"{maindict['save_dir']}/{model_id}_epoch_{args.load_weight}.pth"

    utils.print_dict('trainer', traindict, log_file_path)
    utils.print_dict('loss', lossdict, log_file_path)
    utils.print_dict('data', data, log_file_path)
    utils.print_dict('main', maindict, log_file_path)

    print('begin ------- check param dict -------- ')
    check_param_dict.check_traindict(traindict)
    check_param_dict.check_datadict(data)
    print('end   ------- check param dict -------- ')

    data_set = my_data(data["train_file"], data["valid_file"], data["test_file"],
                       traindict["tau_long"], traindict["window_sliding"], traindict["tau_traj_len"],
                       data["train_pts"], data["valid_pts"], data["test_pts"])
    loader = data_loader(data_set, data["batch_size"])

    train = trainer(traindict, lossdict, log_file=log_file_path)

    start_epoch, best_v_loss = train.load_models()

    print('------- initial learning configurations -------- ')
    # train.verbose(0, 'init_config')     # TODO what is this for??
    train.loss_obj.clear()
    print(f'------- start from epoch {start_epoch} valid loss {best_v_loss} ------ ')

    for e in range(start_epoch, maindict["end_epoch"]):

        cntr = 0
        for qpl_input, qpl_label in loader.train_loader:

            mydevice.load(qpl_input)
            q_traj, p_traj, q_label, p_label, l_init = utils.pack_data(qpl_input, qpl_label)

            train.one_step(q_traj, p_traj, q_label, p_label, l_init)
            cntr += 1
            if cntr % 10 == 0:
                print('.', end='', flush=True)

        print(cntr, 'batches \n')

        if e % maindict["verb"] == 0:
            _ = train.verbose(e, 'train')
            print('time use for ', maindict["verb"], 'epoch is: ', end='')
            system_logs.record_time_usage(e)

        if e % maindict["ckpt_interval"] == 0:
            filename = f"./{maindict['save_dir']}/{model_id}_epoch_{e:d}.pth"
            print('saving file to ', filename)
            train.checkpoint(filename, e, sys.maxsize)

        if e % maindict["val_interval"] == 0:
            train.loss_obj.clear()
            for qpl_input, qpl_label in loader.val_loader:
                q_traj, p_traj, q_label, p_label, l_init = utils.pack_data(qpl_input, qpl_label)
                train.eval(q_traj, p_traj, q_label, p_label, l_init)
            val_loss = train.verbose(e, 'eval')
            if val_loss < best_v_loss:
                best_v_loss = val_loss
                train.checkpoint(f"./{maindict['save_dir']}/{model_id}_best.pth", e, val_loss)

    system_logs.print_end_logs()


if __name__ == '__main__':
    yaml_config_path = 'default_config.yaml'
    with open(yaml_config_path, 'r') as f:
        default_args = yaml.load(f, Loader=yaml.Loader)
    overridden_argv = utils.check_arg_changes(sys.argv, default_args)
    main_args = utils.get_args(default_args)

    ignore_list = ['model_id', 'batch_size', 'load_weight', 'end_epoch', 'poly_deg', 'window_sliding']
    overridden_argv = [k for k in overridden_argv if k not in ignore_list]
    if len(overridden_argv) == 0:
        main_model_name = 'vanilla'
    else:
        main_model_name = '-'.join([f'{k}={getattr(main_args, k)}' for k in sorted(overridden_argv)])
    print(main_model_name)
    main(main_args, main_model_name)
