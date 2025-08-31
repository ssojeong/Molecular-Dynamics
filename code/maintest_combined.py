import torch
import time
import sys
import yaml

from ML.trainer.trainer          import trainer
from ML.predicter.predicter      import predicter
from utils                       import check_param_dict
from utils                       import utils
from utils.system_logs           import system_logs
from utils.mydevice              import mydevice
from data_loader.data_loader import data_loader
from data_loader.data_loader import my_data
import numpy  as np

# python maintest_combined.py api0lw8421ew1repw10poly1lowrho transformer_type gnn_identity mlp_type
# 16 0.035 0.46 0.1 8 809 1000 6 0.2 0 0,1/8,0,1/4,0,1/2,0,1 600000 g 1


def main():
    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    torch.set_default_dtype(torch.float64)

    torch.manual_seed(34952)

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
    gamma = 10
    temp = 300

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
                 "ml_steps": 100
                 }

    lossdict = { "polynomial_degree": 4,
                 "rthrsh": 0.7,
                 "e_weight": 1,
                 "reg_weight": 10}

    data = {"train_file": f'../../Data/LLUF/300k_gap10_train.pt',
            "valid_file": f'../../Data/LLUF/300k_gap10_valid.pt',
            "test_file" : f'../../Data/LLUF/300k_gap10_valid.pt',
            "train_pts" : args['dpt_train'],
            "valid_pts" : args['dpt_valid'],
            "test_pts"  : 100,
            "batch_size": args['batch_size'],
            "window_sliding": window_sliding}
    
    maindict = {"save_dir": f'../../SavedModel/LLUF/{model_name}',
                 "nitr": nitr,  # for check md trajectories
                 "tau_short": 0.002}

    traindict['loadfile'] = f"{maindict['save_dir']}/{model_id}_{10:06d}.pth"

    utils.print_dict('data', data)

    print(traindict)
    print(maindict)

    tau_traj_len = traindict["tau_traj_len"]
    tau_long = traindict["tau_long"]

    traj_len_prep = round(tau_traj_len / tau_long, 4) - 1  # e.g. tau_traj_len=4*2 , tau_traj_prep = 8 - 2

    # print('traj len prep ', traj_len_prep, 't max', traindict['ml_steps']*tau_long, 'predict ml step ', traindict['ml_steps'])

    data_set = my_data(data["train_file"], data["valid_file"], data["test_file"],
                       traindict["tau_long"], traindict["window_sliding"], traindict["tau_traj_len"],
                       data["train_pts"], data["valid_pts"], data["test_pts"])

    loader = data_loader(data_set, data["batch_size"])

    train = trainer(traindict, lossdict)
    train.load_models()

    train.mlvv.eval()
    
    predict = predicter(train.prepare_data_obj, train.mlvv)

    with torch.no_grad():

        cntr = 0

        for qpl_input, qpl_label in loader.test_loader:

            mydevice.load(qpl_input)
            q_traj, p_traj, q_label, p_label, l_init = utils.pack_data(qpl_input, qpl_label)

            q_input_list, p_input_list, q_cur, p_cur = predict.prepare_input_list(q_traj, p_traj, l_init)
            qpl_in = torch.unsqueeze(torch.stack((q_cur, p_cur, l_init), dim=1), dim=2)   # use concat inital state

            qpl_batch = []
            start_time = time.time()

            for t in range(traindict['ml_steps']):

                print('====== t=', round(traj_len_prep + t * tau_long, 3), ' window sliding ', t+1,
                      't=', round((t+1) * tau_long + traj_len_prep, 3), flush=True)

                q_input_list, p_input_list, q_predict, p_predict, l_init = predict.eval(q_input_list,p_input_list, q_cur,p_cur,l_init, t+1, gamma, temp, tau_long)

                qpl_list = torch.stack((q_predict, p_predict, l_init), dim=1)

                # if (t + 1) % traindict['append_strike'] == 0:
                qpl_batch.append(qpl_list)

                q_cur = q_predict
                p_cur = p_predict

            sec = time.time() - start_time
            # sec = sec / maindict["nitr"]
            # mins, sec = divmod(sec, 60)
            print("{} nitr --- {:.03f} sec ---".format(traindict['ml_steps'], sec))
            print("samples {}, one forward step timing --- {:.03f} sec ---".format(data["batch_size"], sec/traindict['ml_steps']))

            qpl_batch = torch.stack(qpl_batch, dim=2)   # shape [nsamples,3, traj_len, nparticles,dim]
            # qpl_batch [nsamples,3,traj,nparticles,dim]

            print('====== load no batch ', cntr, '==== shape ', qpl_in.shape,qpl_batch.shape)
            qpl_batch_cat = torch.cat((qpl_in, qpl_batch), dim=2)   # stack traj initial + window-sliding

            tmp_filename = maindict["save_dir"] + str(traindict['tau_long']) + f'_id{cntr}.pt'
            print('saved qpl list shape', qpl_batch_cat.shape)
            torch.save({'qpl_trajectory': qpl_batch_cat,
                        'tau_short': maindict['tau_short'],
                        'tau_long': traindict["tau_long"]}, tmp_filename)
            # if i == 3*step: quit()
            cntr += 1
            # if cntr%10==0: print('.', end='', flush=True)
 
# system_logs.print_end_logs()


if __name__ == '__main__':
    yaml_config_path = 'config.yaml'
    with open(yaml_config_path, 'r') as f:
        args = yaml.load(f, Loader=yaml.Loader)
    main()

