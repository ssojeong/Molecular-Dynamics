import torch
import time
import sys
import yaml

import matplotlib.pyplot as plt

from ML.trainer.trainer          import trainer
from ML.predicter.predicter      import predicter
from ML.predicter.rdf_H2O        import count2rdf, q2dis
from ML.trainer.loss             import loss
from utils                       import utils
from utils.system_logs           import system_logs
from utils.mydevice              import mydevice
from data_loader.data_loader import data_loader
from data_loader.data_loader import my_data


def main():
    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    torch.set_default_dtype(torch.float64)

    torch.manual_seed(34952)

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
    gamma = 0
    temp = 300
    # ==========================
    nnodes = [args.pwnet_dim] * args.pwnet_layer

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
                 "ml_steps"     : 8,
                 "append_strike": 1
                 }

    lossdict = {"polynomial_degree": poly_deg,
                "rthrsh"           : 0.7,
                "e_weight"         : ew,
                "reg_weight"       : repw}

    data = {"train_file": f'../../Data/LLUF/300k_100ktraj_gap{gap}_train.pt',
            "valid_file": f'../../Data/LLUF/300k_100ktraj_gap{gap}_valid.pt',
            "test_file" : f'../../Data/LLUF/300k_100ktraj_gap{gap}_train.pt',
            "train_pts" : args.dpt_train,
            "valid_pts" : args.dpt_valid,
            "test_pts"  : 1280,
            "batch_size": args.batch_size * 2,
            "window_sliding": window_sliding}

    maindict = {"end_epoch"       : args.end_epoch,
                "save_dir"        : f'../../SavedModel/LLUF/',
                "tau_short"       : 1e-4,
                "nitr"            : nitr,  # for check md trajectories
                "append_strike"   : nitr,  # for check md trajectories
                "ckpt_interval"   : 1,     # for check pointing
                "val_interval"    : 1,     # no use of valid for now
                "verb"            : 1}   # period for printing out losses

    traindict['loadfile'] = '../../SavedModel/LLUF/0_000027.pth'
    utils.print_dict('data', data)

    print(traindict)
    print(maindict)

    tau_traj_len = traindict["tau_traj_len"]
    tau_long = traindict["tau_long"]

    traj_len_prep = round(tau_traj_len / tau_long, 4) - 1  # e.g. tau_traj_len=4*2 , tau_traj_prep = 8 - 2

    data_set = my_data(data["train_file"], data["valid_file"], data["test_file"],
                       traindict["tau_long"], traindict["window_sliding"], traindict["tau_traj_len"],
                       data["train_pts"], data["valid_pts"], data["test_pts"])

    loader = data_loader(data_set, data["batch_size"])

    train = trainer(traindict, lossdict)
    train.load_models()

    train.mlvv.eval()
    
    predict = predicter(train.prepare_data_obj, train.mlvv)
    loss_obj = loss(lossdict["polynomial_degree"],
                    lossdict["rthrsh"],
                    lossdict["e_weight"],
                    lossdict["reg_weight"],
                    traindict["window_sliding"])  # remove eweight in loss

    with torch.no_grad():

        cntr = 0
        qpl_epoch = []
        q_rmse_epoch = []
        p_rmse_epoch = []

        for qpl_input, qpl_label in loader.test_loader:

            mydevice.load(qpl_input)
            q_traj, p_traj, q_label, p_label, l_init = utils.pack_data(qpl_input, qpl_label)
            print('q traj shape', q_traj.shape, 'q label shape', q_label.shape, 'qpl label', qpl_label.shape)

            q_input_list, p_input_list, q_cur, p_cur = predict.prepare_input_list(q_traj, p_traj, l_init)
            qpl_in = torch.unsqueeze(torch.stack((q_cur, p_cur, l_init), dim=1), dim=2)   # use concat initial state

            qpl_batch = []
            q_rmse_batch = torch.zeros(traindict['ml_steps'], requires_grad=False)
            p_rmse_batch = torch.zeros(traindict['ml_steps'], requires_grad=False)
            start_time = time.time()
            for t in range(traindict['ml_steps']):

                # print('==== t=', round(traj_len_prep + t * tau_long, 3), ' window sliding ', t+1,
                #       't=', round((t+1) * tau_long + traj_len_prep, 3), flush=True)

                q_input_list, p_input_list, q_predict, p_predict, l_init = predict.eval(q_input_list, p_input_list, q_cur, p_cur, l_init, t+1, gamma, temp, tau_long)

                qpl_list = torch.stack((q_predict, p_predict, l_init), dim=1)

                if (t + 1) % traindict['append_strike'] == 0:
                    qpl_batch.append(qpl_list)
                    # print('qpl length', len(qpl_batch))

                # q_noise = 0.1 * torch.rand(q_predict.shape, device=q_predict.device)
                # print(torch.std(q_noise))
                # q_predict += q_noise
                q_cur = q_predict
                p_cur = p_predict

                q_rmse_batch[t] += loss_obj.q_RMSE_loss(q_predict, q_label[:, t], l_init).mean().item()
                p_rmse_batch[t] += loss_obj.p_RMSE_loss(p_predict, p_label[:, t]).mean().item()
                # quit()
            # print(q_rmse_batch, p_rmse_batch)
            sec = time.time() - start_time
            # sec = sec / maindict["nitr"]
            # mins, sec = divmod(sec, 60)
            # print(f"{traindict['ml_steps']} nitr --- {sec:.03f} sec ---")
            print(f"samples {data['batch_size']}, one forward step timing --- {sec/traindict['ml_steps']:.03f} sec ---")

            qpl_batch = torch.stack(qpl_batch, dim=2)   # shape [nsamples,3, traj_len, nparticles,dim]
            # qpl_batch [nsamples,3,traj,nparticles,dim]

            print('==== load batch ', cntr, '==== shape ', qpl_in.shape, qpl_batch.shape)
            qpl_batch_cat = torch.cat((qpl_in, qpl_batch), dim=2)   # stack traj initial + window-sliding

            # print('batch', cntr, 'saved qpl list shape', qpl_batch_cat.shape)

            qpl_epoch.append(qpl_batch_cat)
            q_rmse_epoch.append(q_rmse_batch)
            p_rmse_epoch.append(p_rmse_batch)
            cntr += 1

        qpl_epoch = torch.cat(qpl_epoch)
        q_rmse_epoch = torch.stack(q_rmse_epoch).mean(dim=0)
        p_rmse_epoch = torch.stack(p_rmse_epoch).mean(dim=0)
        print('qpl epoch', qpl_epoch.shape)
        print('q rmse', q_rmse_epoch, 'p rmse', p_rmse_epoch)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
        ax1.plot(range(len(q_rmse_epoch)), q_rmse_epoch, label='q rmse')
        ax2.plot(range(len(p_rmse_epoch)), p_rmse_epoch, label='p rmse')
        plt.suptitle('1/8-1/7-...-1')
        ax1.legend()
        ax2.legend()
        ax1.grid()
        ax2.grid()
        plt.show()
        rho = 8 / 2.2 ** 3
        #
        for i in range(traindict['ml_steps']):
            counts, bin_edges = q2dis(qpl_epoch[:, 0, i+1], num_mol=8, n_bins=200, r_min=0, r_max=2, box_size=2.2)
            grbin = count2rdf(counts, bin_edges, rho, n_sample=qpl_epoch.size(0), num_mol=8)
            # print(grbin)
            data = {'counts': counts,
                    'gr': torch.tensor(grbin),
                    'edge_centers': (bin_edges[:-1] + bin_edges[1:]) / 2}
            torch.save(data, f'train_1-1-ws{i}.pt')


if __name__ == '__main__':
    yaml_config_path = 'default_config.yaml'
    with open(yaml_config_path, 'r') as f:
        default_args = yaml.load(f, Loader=yaml.Loader)
    overridden_argv = utils.check_arg_changes(sys.argv, default_args)
    args = utils.get_args(default_args)
    main()

