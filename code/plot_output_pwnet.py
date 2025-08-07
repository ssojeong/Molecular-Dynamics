import torch
import sys
import math
import numpy as np
from utils.system_logs import system_logs
from utils.mydevice import mydevice
from utils          import utils
from ML.trainer.trainer import trainer
import matplotlib.pyplot as plt

def plot_pw_graph(r, features, ncols, fig_filename):

        fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(10, 6))
        for i in range(ncols):
            ax[i].plot(r, features[:,i],'o',label='feature {}'.format(i+1))
            #ax[1].set_title('feature 2')

        for j in range(ncols):
            ax[j].grid()
            ax[j].set_xticks([1,2,3])
            ax[j].tick_params(axis='x', labelsize=15)
            ax[j].tick_params(axis='y', labelsize=15)
            #ax[j].legend(loc='upper right', fontsize=10)
            #ax[j].set_ylim([-2.5, 3.5])
            ax[j].legend(fontsize=15)
            ax[j].set_xlabel('r',fontsize=15)

        #fig.suptitle('6 grids centered of particle {} and interacted with particle {}, {}'.format(i+1,k+1, fig_filename), fontsize=16)
        fig.suptitle('{}'.format( fig_filename), fontsize=16)
        plt.tight_layout()
        plt.show()
        #fig.savefig(fig_filename + '_par{}_{}.pdf'.format(i+1,k+1), bbox_inches='tight', dpi=200)

def l_max_distance(l_list):
    boxsize = torch.mean(l_list)
    L_h = boxsize / 2.
    q_max = math.sqrt(L_h * L_h + L_h * L_h)
    print('boxsize', boxsize.item(), 'maximum distance dq = {:.2f}, dq^2 = {:.2f}'.format(q_max, q_max * q_max))
    return boxsize, q_max

if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES='' python plot_output_pwnet.py results20240706/traj_len08ws08tau0.1ngrid6api0lw8421ew1repw10poly1lg_dpt800000/mbpw000485.pth
    # 'Trained pair-wise model on liquid+gas region; mb485; qloss 0.0038; ploss 0.020'

    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    torch.set_default_dtype(torch.float64)

    torch.manual_seed(34952)
    np.random.seed(34952)

    argv = sys.argv
    if len(argv) != 3:
        print('usage <programe> <loadfile> <title>' )
        quit()

    loadfile = argv[1]
    title = argv[2]


    traindict = {"loadfile"     : loadfile,  # to load previously trained model
                 "net_nnodes"   : 128,   # number of nodes in neural nets
                 "pw4mb_nnodes" : 128,   # number of nodes in neural nets
                 "pw_output_dim" : 3,
                 "init_weights"   : 'tanh',
                 "optimizer" : 'Adam',
                 "single_particle_net_type" : 'transformer_type',             # mlp_type       transformer_type    transformer_type
                 "multi_particle_net_type"  : 'gnn_identity',         # gnn_identity   gnn_identity        gnn_transformer_type
                 "readout_step_net_type" : 'mlp_type',             # mlp_identity   mlp_type            mlp_type
                 "n_encoder_layers" : 2,
                 "n_gnn_layers" : 2,
                 "edge_attention" : True,
                 "d_model"      : 256,
                 "nhead"        : 8,
                 "net_dropout"  : 0.0,    # 1: all zero ; 0 : not dropout ; 0.9 : 90% zero
                 "grad_clip"    : 0.5,    # clamp the gradient for neural net parameters
                 "tau_traj_len" : 8*0.1,  # n evaluations in integrator
                 "tau_long"     : 0.1,
                 "loss_weights": '0,1/8,0,1/4,0,1/2,0,1',
                 "window_sliding" : 8,      # number of times to do integration before cal the loss
                 "ngrids"       : 12,      # for multibody interactions
                 "b_list"       : [0.2], #[0.4],
                 "a_list"       : [0],#[np.pi/8],
                 "maxlr"        : 1e-5,   # starting learning rate # HK
                 "tau_init"     : 1,      # starting learning rate
                 }

    loss_weights = traindict['loss_weights'].split(',')

    window_sliding_list = [1,2,3,4,5,7,8,9,10,12,14,16]

    if traindict["window_sliding"] not in window_sliding_list:
        print('window_sliding is not valid, need ',window_sliding_list)
        quit()

    lossdict = { "polynomial_degree" : 4,
                 "rthrsh"       : 0.7,
                 "e_weight"    : 1,
                 "reg_weight"    : 10}

    data = { "test_file" : '../data_sets/gen_by_MD/3d/n32lt0.1stpstraj18_l_dpt45000.pt',
             "test_pts"  : 1000,
             "batch_size": 1000,
             "window_sliding"   : traindict["window_sliding"] }

    maindict = { "start_epoch"     : 0,
                 "end_epoch"       : 4,
                 "save_dir"        : './results/',
                 "nitr"            : 1000, # for check md trajectories
                 "tau_short"       : 1e-4,
                 "append_strike"   : 1000, # for check md trajectories
                 "ckpt_interval"   : 2, # for check pointing
                 "val_interval"    : 1, # no use of valid for now
                 "verb"            : 1  } # peroid for printing out losses


    utils.print_dict('trainer', traindict)
    utils.print_dict('data', data)
    utils.print_dict('main', maindict)

    r = np.arange(0,3,0.01)

    train = trainer(traindict, lossdict)
    r = torch.tensor(r)
    mydevice.load(r)
    #print(r.is_cuda)
    r = torch.unsqueeze(r,dim=1)

    pair_wise = train.mlvv.prepare_data.net(r)
    pw_output_dim = traindict["pw_output_dim"]

    #plot_pw_graph(r.detach().cpu().numpy(), pair_wise.detach().cpu().numpy(), pw_output_dim, 'b4training')

    train.load_models()
    train.mlvv.eval()

    pair_wise = train.mlvv.prepare_data.net(r)

    plot_pw_graph(r.detach().cpu().numpy(), pair_wise.detach().cpu().numpy(), pw_output_dim, title)



