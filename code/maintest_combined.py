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

# python maintest_combined.py api0lw8421ew1repw10poly1lowrho transformer_type gnn_identity mlp_type
# 16 0.035 0.46 0.1 8 809 1000 6 0.2 0 0,1/8,0,1/4,0,1/2,0,1 600000 g 1
if __name__=='__main__':

    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    torch.set_default_dtype(torch.float64)

    torch.manual_seed(34952)

    argv = sys.argv
    if len(argv) != 19:
        print('usage <programe> <net_type> <single_parnet_type> <multi_parnet_type> <readout_net_type> '
              '<npar> <rho> <temp> <tau_long> <window_sliding> <saved model> <nitr> <ngrid> <b> <a> <loss weight> <region> <gamma> ' )
        quit()

    net_type = argv[1]
    single_parnet_type = argv[2]
    multi_parnet_type = argv[3]
    readout_net_type = argv[4]
    npar = argv[5]
    rho = argv[6]
    temp = float(argv[7])
    tau_long = float(argv[8])
    window_sliding = int(argv[9])
    num_saved = str(argv[10])
    nitr = int(argv[11])
    ngrid = int(argv[12])
    b = argv[13].split(',')
    a = argv[14].split(',')
    loss_weights = argv[15].split(',')
    dpt = int(argv[16])
    region = argv[17]
    gamma = float(argv[18])

    if (gamma == 0.0) or (gamma == 1.0) or (gamma == 10.0) or (gamma == 20.0):
        print('gamma {} float to int .... '.format(gamma))
        gamma = int(gamma)

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


    traindict = {"loadfile": './results/traj_len08ws0{}tau{}ngrid{}{}_dpt{}/mbpw000{}.pth'.format(
                                  window_sliding,tau_long,ngrid, net_type,dpt,num_saved),  # to load previously trained model
                 "net_nnodes": 128,  # number of nodes in neural nets for force
                 "pw4mb_nnodes" : 128,
                 "pw_output_dim" : 2,
                 "init_weights"   : 'tanh',
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
                 "loss_weights": loss_weights,
                  "window_sliding": window_sliding,  # number of times to do integration before cal the loss
                  "ngrids": ngrid,  # 6*len(b_list)
                  "b_list": b,  # # grid lattice constant for multibody interactions
                  "a_list": a,  # [np.pi/8], #
                  "ml_steps" : 10000,
                  "append_strike" : 10,
                  "maxlr"    : 1e-5,
                  "tau_init": 1 , # starting learning rate
                 "gamma" : gamma
                 }

    lossdict = { "polynomial_degree" : 4,
                 "rthrsh"       : 0.7,
                 "e_weight"    : 1,
                 "reg_weight"    : 10}

    data = { "train_file": '../data_sets/gen_by_MD/noML-metric-st1e-4every0.1t100/n{}rho{}T{}'.format(npar,rho,temp)
             + '/n{}rho{}T{}.pt'.format(npar,rho,temp),
             "valid_file": '../data_sets/gen_by_MD/noML-metric-st1e-4every0.1t100/n{}rho{}T{}'.format(npar,rho,temp)
             + '/n{}rho{}T{}.pt'.format(npar,rho,temp),
             "test_file" : '../data_sets/gen_by_MD/noML-metric-st1e-4every0.1t100/n{}rho{}T{}'.format(npar,rho,temp)
             + '/n{}rho{}T{}.pt'.format(npar,rho,temp), 
             "train_pts" : 10,
             "vald_pts"  : 10,
             "test_pts"  : 1000,
             "batch_size": 20}
    
    maindict = {
                 "save_dir"     : '../data_sets/gen_by_ML/lt{}dpt{}_{}/n{}rho{}T{}/'.format(tau_long,dpt,
                                   region,npar,rho,temp)  + "pred_n{}len08ws08gamma{}mb{}_tau".format(npar,gamma,num_saved),
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
    #check_param_dict.check_testdict(maindict)
    #assert ( traindict["tau_traj_len"] == maindict['tau_max'] ), "not match tau traj len and tau_max"
    print('end   ------- check param dict -------- ')

    tau_traj_len = traindict["tau_traj_len"]
    tau_long = traindict["tau_long"]

    traj_len_prep = round(tau_traj_len / tau_long , 4) - 1  # e.g. tau_traj_len=4*2 , tau_traj_prep = 8 - 2

    print('traj len prep ', traj_len_prep, 't max', traindict['ml_steps']*tau_long , 'predict ml step ', traindict['ml_steps'])

    data_set = my_data(data["train_file"],data["valid_file"],data["test_file"],
                       traindict["tau_long"],traindict["window_sliding"],traindict["tau_traj_len"],
                       data["train_pts"],data["vald_pts"],data["test_pts"])

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

              q_input_list,p_input_list, q_predict, p_predict, l_init = predict.eval(q_input_list,p_input_list,q_cur,p_cur,l_init, t+1, gamma, temp, tau_long)

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


