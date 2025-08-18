import torch
import time
import sys
from ML.trainer.trainer          import trainer
from ML.predicter.predicter      import predicter
from utils                       import check_param_dict
from utils                       import utils
from utils.system_logs           import system_logs
from utils.mydevice              import mydevice
from data_loader.data_loader import data_loader
from data_loader.data_loader import my_data
import numpy  as np
import yaml

def main():

    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(34952)

    net_type = args['net_type']
    single_parnet_type = args['single_parnet_type']
    multi_parnet_type = args['multi_parnet_type']
    readout_net_type = args['readout_net_type']
    npar = args['npar']
    rho = args['rho']
    temp = args['temp']
    tau_long = args['tau_long']
    saved_pair_steps = args['saved_pair_steps']
    window_sliding = args['window_sliding']
    num_saved = args['num_saved']
    nitr = args['nitr']
    ngrid = args['ngrid']
    b = args['b']
    a = args['a']
    loss_weights = args['loss_weights']
    ml_steps = args['ml_steps']
    append_strike = args['append_strike']
    dpt = args['dpt']
    region = args['region']
    gamma = args['gamma']


    traindict = {"loadfile": './results/traj_len08ws0{}tau{}ngrid{}{}_dpt{}/mbpw000{}.pth'.format(
                                  window_sliding,tau_long,ngrid, net_type,dpt,num_saved),  # to load previously trained model
                 "net_nnodes": 128,  # number of nodes in neural nets for force
                 "pw4mb_nnodes" : 128,
                 "pw_output_dim" : 3, # 20250812: change from 2D to 3D, psi
                 "init_weights"   : 'relu',
                 "optimizer" : 'Adam',
                 "single_particle_net_type" : single_parnet_type,
                 "multi_particle_net_type"  : multi_parnet_type,
                 "readout_step_net_type"    : readout_net_type,
                 "n_encoder_layers": 2,  # ---LW
                 "n_gnn_layers" : 2,
                  "edge_attention" : True,
                  "d_model": 256,  # ---LW
                  "nhead": 8,  # ---LW
                  "net_dropout": 0.0,  # 1: all zero ; 0 : not dropout ; 0.9 : 90% zero
                  "grad_clip": 0.5,  # clamp the gradient for neural net parameters
                  "tau_traj_len": 8 * tau_long,  # n evaluations in integrator
                  "tau_long": tau_long,
                  "saved_pair_steps": saved_pair_steps,
                 "loss_weights": loss_weights,
                  "window_sliding": window_sliding,  # number of times to do integration before cal the loss
                  "ngrids": ngrid,  # 6*len(b_list)
                  "b_list": b,  # # grid lattice constant for multibody interactions
                  "a_list": a,  # [np.pi/8], #
                  "ml_steps" : ml_steps, # ai 0.1 -> 10000; ai 0.02 -> 50000; ai 0.01 -> 100000
                  "append_strike" : append_strike, # ai 0.1 -> 10; ai 0.02 -> 50; ai 0.01 -> 100
                  "maxlr"    : 1e-5,
                  "tau_init": 1 , # starting learning rate
                 "gamma" : gamma
                 }

    lossdict = { "polynomial_degree" : 4,
                 "rthrsh"       : 0.7,
                 "e_weight"    : 1,
                 "reg_weight"    : 10}

    data = { "train_file": '../data_sets/gen_by_MD/3d/noML-metric-st1e-4every0.1t8/n{}rho{}T{}'.format(npar,rho,temp)
             + '/n{}rho{}T{}.pt'.format(npar,rho,temp),
             "valid_file": '../data_sets/gen_by_MD/3d/noML-metric-st1e-4every0.1t8/n{}rho{}T{}'.format(npar,rho,temp)
             + '/n{}rho{}T{}.pt'.format(npar,rho,temp),
             "test_file" : '../data_sets/gen_by_MD/3d/noML-metric-st1e-4every0.1t8/n{}rho{}T{}'.format(npar,rho,temp)
             + '/n{}rho{}T{}.pt'.format(npar,rho,temp), 
             "train_pts" : 10,
             "vald_pts"  : 10,
             "test_pts"  : 1000,
             "batch_size": 100}
    
    maindict = {
                 "save_dir"     : '../data_sets/gen_by_ML/3d/lt{}dpt{}_{}/n{}rho{}T{}/'.format(tau_long,dpt, region,npar,rho,temp)
                                  + "pred_n{}len08ws08gamma{}LUF{}_tau".format(npar,gamma,num_saved),
                 "nitr"         : nitr,  # for check md trajectories
                 "tau_short"    : 1e-4,
                 "append_strike": nitr, # for check md trajectories
               }

    utils.print_dict('data',data)

    print(traindict)
    print(maindict)

    print('begin ------- check param dict -------- ')
    check_param_dict.check_maindict(traindict)
    check_param_dict.check_traindict(maindict,traindict["tau_long"])
    # check_param_dict.check_testdict(maindict)

    print('end   ------- check param dict -------- ')
    tau_traj_len = traindict["tau_traj_len"]
    tau_long = traindict["tau_long"]

    traj_len_prep = round(tau_traj_len / tau_long) * saved_pair_steps - saved_pair_steps

    print('traj len prep ', traj_len_prep, 't max', traindict['ml_steps']*tau_long , 'predict ml step ', traindict['ml_steps'])

    data_set = my_data(data["train_file"],data["valid_file"],data["test_file"],
                       traindict["tau_long"],traindict["window_sliding"],traindict["saved_pair_steps"],
                       traindict["tau_traj_len"],data["train_pts"],data["vald_pts"],data["test_pts"])  # 20250912

    loader = data_loader(data_set, data["batch_size"])

    train = trainer(traindict,lossdict)
    train.load_models()

    train.mlvv.eval()
    
    predict = predicter(train.prepare_data_obj, train.mlvv)

    with torch.no_grad():

      cntr = 0 
    
      for qpl_input,qpl_label in loader.test_loader:

          mydevice.load(qpl_input)
          q_traj,p_traj,q_label,p_label,l_init = utils.pack_data(qpl_input, qpl_label)

          q_input_list,p_input_list,q_cur,p_cur = predict.prepare_input_list(q_traj,p_traj,l_init)
          qpl_in = torch.unsqueeze(torch.stack((q_cur,p_cur,l_init),dim=1),dim=2) # use concat inital state

          qpl_batch = []
          start_time = time.time()

          for t in range(traindict['ml_steps']):

              print('====== t=',round(traj_len_prep + t * tau_long,3),' window sliding ', t+1,
                    't=', round((t+1) * tau_long + traj_len_prep,3), flush=True)
              #print('q',q_next.shape,'p', p_next.shape,'l', l_next.shape)

              q_input_list,p_input_list, q_predict, p_predict, l_init = predict.eval(q_input_list,p_input_list,q_cur,p_cur,l_init, t+1, gamma, temp)

              #if (t+1)%20 == 0: quit()
              qpl_list = torch.stack((q_predict, p_predict, l_init), dim=1)

              if (t + 1) % traindict['append_strike'] == 0:
                qpl_batch.append(qpl_list)

              q_cur = q_predict 
              p_cur = p_predict 

              #if t >= 1: quit()
              # print('t= ',t, 'CPU memory % used:', psutil.virtual_memory()[2], '\n')
  
          sec = time.time() - start_time
          #sec = sec / maindict["nitr"]
          #mins, sec = divmod(sec, 60)
          print("{} nitr --- {:.03f} sec ---".format(traindict['ml_steps'],sec))
          print("samples {}, one forward step timing --- {:.03f} sec ---".format(data["batch_size"], sec/traindict['ml_steps']))

          qpl_batch = torch.stack(qpl_batch, dim=2) # shape [nsamples,3, traj_len, nparticles,dim]
          # qpl_batch [nsamples,3,traj,nparticles,dim]

          print('====== load no batch ' , cntr, '==== shape ', qpl_in.shape,qpl_batch.shape)
          qpl_batch_cat = torch.cat((qpl_in,qpl_batch), dim=2) # stack traj inital + window-sliding
 
          tmp_filename = maindict["save_dir"] + str(traindict['tau_long']) + f'_id{cntr}.pt'     
          print('saved qpl list shape', qpl_batch_cat.shape)
          torch.save({'qpl_trajectory': qpl_batch_cat, 'tau_short':maindict['tau_short'], 'tau_long' : traindict["tau_long"]},tmp_filename )
          # if i == 3*step: quit()
          cntr += 1
          #if cntr%10==0: print('.', end='', flush=True)

  
    #   qpl_list2 = torch.cat(qpl_list_sample, dim=0)
      # qpl_list [nsamples,3,traj,nparticles,dim]
 
    system_logs.print_end_logs()

if __name__=='__main__':
    yaml_config_path = 'maintest_config.yaml'
    with open(yaml_config_path, 'r') as f:
        args = yaml.load(f, Loader=yaml.Loader)
    main()



