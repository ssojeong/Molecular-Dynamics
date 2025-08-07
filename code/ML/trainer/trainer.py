import torch.nn as nn
import torch.optim as optim
from utils.mydevice import mydevice
from ML.trainer.loss import loss

from ML.networks.PWNet import PWNet
from ML.networks.SingleParticleMLPNet import SingleParticleMLPNet
from ML.networks.SingleParticleTransformerNet import SingleParticleTransformerNet
from ML.networks.MultiParticlesGraphNet import MultiParticlesGraphNet
from ML.networks.ReadoutStepMLPNet import ReadoutStepMLPNet

#from ML.LLUF.HexGrids import HexGrids # comment tentative, refector later after fix isocahedron
from ML.LLUF.isocahedron import isocahedron # 20250803: change Hexgrids to isocahedron
from ML.LLUF.SingleGrid import SingleGrid
from ML.LLUF.PrepareData import PrepareData
from ML.LLUF.HalfStepUpdate import HalfStepUpdate
#from MD.LLUF_MD import LLUF_MD
from MD.LLUF_Lengavin import LLUF_Lengavin
from utils.checkpoint import checkpoint

class trainer:

    def __init__(self,train_dict,loss_dict):

        self.train_dict = train_dict

        # lj is use for calculating regularization on replusive force and conservation of energy
        prepare_data_net, single_particle_net_list, multi_particle_gnn_net_list, readout_step_mlp_net_list = self.net_builder(train_dict)
        assert train_dict["pw_output_dim"]==3,'current implementation is for 3D system only' #20250807

        if train_dict["ngrids"] > 1:
            # print('create HexGrid object ....') # 20250803: change Hexgrids to isocahedron
            # grid_object = HexGrids(train_dict["b_list"],train_dict["a_list"])
            # 20250807
            assert train_dict['ngrids'] == 12, 'ngrid should be 12 for isocahedron.'
            print('create isocahedron object ....') # 20250803: change Hexgrids to isocahedron
            grid_object = isocahedron(train_dict["b_list"], train_dict["a_list"])
        else:
            print('create SingleGrid object ....this function is defunct, quiting')
            assert False,'SingleGrid implemention for test is now defunct'
            grid_object = SingleGrid()

        self.prepare_data_obj = PrepareData(prepare_data_net, grid_object)

        self.LLUF_update_p_obj = HalfStepUpdate(self.prepare_data_obj, single_particle_net_list[0],
                                multi_particle_gnn_net_list[0], readout_step_mlp_net_list[0],train_dict["tau_init"])
        self.LLUF_update_q_obj = HalfStepUpdate(self.prepare_data_obj, single_particle_net_list[1],
                                multi_particle_gnn_net_list[1], readout_step_mlp_net_list[1],train_dict["tau_init"])

        self.mlvv = LLUF_Lengavin(self.prepare_data_obj, self.LLUF_update_p_obj,self.LLUF_update_q_obj)
        # # 3 tau here. tau[0] for update p, tau[1],tau[2] for update q

        self.tau_params = [self.LLUF_update_p_obj.tau, self.LLUF_update_q_obj.tau,self.mlvv.tau] # for printing

        if train_dict["optimizer"] == 'SGD' :
            print( "optim given .... sgd.....")
            self.opt = optim.SGD(self.mlvv.parameters(),lr=train_dict["maxlr"])
        elif train_dict["optimizer"] == 'Adam' :
            print( "optim given .... adam ....")
            self.opt = optim.Adam(self.mlvv.parameters(),lr=train_dict["maxlr"])
        elif train_dict["optimizer"] == 'AdamW' :
            print( "optim given .... adamW ....")
            self.opt = optim.AdamW(self.mlvv.parameters(),lr=train_dict["maxlr"])
        else:
            assert (False), 'invalid optimizer given'

        self.weights = train_dict['loss_weights']
        print( "loss weight .... {}...".format(self.weights))

        self.loss_obj = loss(loss_dict["polynomial_degree"],loss_dict["rthrsh"],
                             loss_dict["e_weight"],loss_dict["reg_weight"],train_dict["window_sliding"]) # remove eweight in loss

        self.ckpt = checkpoint(self.mlvv, self.opt)
        self.grad_clip = self.train_dict["grad_clip"]

   # ==========================================================
    def one_step(self,q_traj,p_traj,q_label,p_label,l_init):

        self.opt.zero_grad()

        # need to mask out particle itself interaction at each pw-net
        #self.prepare_data_obj.make_mask(nsamples,nparticles,dim)

        # q_traj,p_traj [traj,nsamples,nparticles,dim]
        q_traj_list = list(q_traj)
        p_traj_list = list(p_traj)

        q_cur = q_traj_list[-1]
        p_cur = p_traj_list[-1]

        q_input_list = []
        p_input_list = []
        # append over trajectory length
        for q,p in zip(q_traj_list,p_traj_list):
            # q,p shape [nsamples,nparticles,dim]
            q_input_list.append(self.prepare_data_obj.prepare_q_feature_input(q,l_init))
            p_input_list.append(self.prepare_data_obj.prepare_p_feature_input(q,p,l_init))


        loss_val = 0

        for ws in range(self.train_dict["window_sliding"]):

            # q_input_list  [(phi0,dq0),(phi1,dq1),(phi2,dq2),...] : tuple inside list
            # phi is function of q at grid point as input for mb-net, dq is input for pw-net
            # p_input_list  [(pi0,dp0),(pi1,dp1),(pi2,dp2),...]
            # pi is momentum at grid point as input for mb-net, dp is input for pw-net
            q_input_list,p_input_list,q_predict,p_predict,l_init = self.mlvv.nsteps(q_input_list,p_input_list,q_cur,p_cur,
                                                     l_init)
    
            loss_val += self.loss_obj.eval(q_predict,p_predict,q_label[:,ws],p_label[:,ws],q_traj_list[-1],p_traj_list[-1],l_init,self.weights[ws])

            q_cur = q_predict
            p_cur = p_predict

        loss_val.backward()

        nn.utils.clip_grad_value_(self.mlvv.parameters(), clip_value=self.grad_clip)

        self.opt.step()
        #print('params', self.opt.param_groups[0]["params"])

    # ==========================================================
    def eval(self,q_traj,p_traj,q_label,p_label,l_init):

        self.mlvv.eval()

        # need to mask out particle itself interaction at each pw-net
        #self.prepare_data_obj.make_mask(nsamples,nparticles,dim)

        q_traj_list = list(q_traj)
        p_traj_list = list(p_traj)

        q_cur = q_traj_list[-1]
        p_cur = p_traj_list[-1]

        q_input_list = []
        p_input_list = []
        for q,p in zip(q_traj_list,p_traj_list):
            q_input_list.append(self.prepare_data_obj.prepare_q_feature_input(q,l_init))
            p_input_list.append(self.prepare_data_obj.prepare_p_feature_input(q,p,l_init))

        loss_val = 0

        for ws in range(self.train_dict["window_sliding"]):

            # q_input_list [phi0,phi1,phi2,...] ; phi is function of q at grid point
            # p_input_list [pi0,pi1,pi2,...] ; pi is momentum at grid point
            q_input_list,p_input_list,q_predict,p_predict,l_init = self.mlvv.nsteps(q_input_list,p_input_list,q_cur,p_cur,
                                                     l_init)

            loss_val += self.loss_obj.eval(q_predict,p_predict,q_label[:,ws],p_label[:,ws],q_traj_list[-1],p_traj_list[-1],l_init,self.weights[ws])

            q_cur = q_predict
            p_cur = p_predict

        self.mlvv.train()
    
    # ==========================================================
    def checkpoint(self,filename):
        self.ckpt.save_checkpoint(filename)
        print('checkpint to file ',filename)
        self.verbose(0,'checkpoint values')

    # ==========================================================
    def load_models(self):
        load_file = self.train_dict["loadfile"]
        if load_file is not None:
            self.ckpt.load_checkpoint(load_file) 

    # ==========================================================
    def verbose(self,e,mode):

        print(e, mode, ' '.join('tau {} : {:2e}'.format(k, tau.item()) for k, tau in enumerate(self.tau_params)))

        if self.tau_params[0].grad is not None:
            print(e, mode, ' '.join('tau grad {} : {}'.format(k, tau.grad) for k, tau in enumerate(self.tau_params)))
        else:
            print('tau grad is None before gradient')

        cur_lr = self.opt.param_groups[0]['lr']
        self.loss_obj.verbose(e,cur_lr,mode)
        self.loss_obj.clear()

        self.LLUF_update_p_obj.f_stat.print(e,  mode + ' p')
        self.LLUF_update_p_obj.f_stat.clear()
        self.LLUF_update_q_obj.f_stat.print(e, mode + ' q')
        self.LLUF_update_q_obj.f_stat.clear()

    # ==========================================================
    def net_builder(self,train_dict):

        ngrids = train_dict["ngrids"]
        pw4mb_dim    = train_dict["pw_output_dim"]
        psi_feature_dim = pw4mb_dim # 20250807
        output_dim =  psi_feature_dim # 20250807 # 20250803 psi features output dim, change 2 to 3.

        tau_traj_len = train_dict["tau_traj_len"]
        tau_long = train_dict["tau_long"]

        factor = round(tau_traj_len/tau_long)
        #input_dim    = 2 * ngrids * dim * factor # each of 6 grids, func of q (f1,f2), p (p1,p2)
        input_dim    =  ngrids * (pw4mb_dim + psi_feature_dim) * factor # 20250807
        # each of 6 grids, func of q (f1,f2), p (p1,p2)

        single_particle_net_list = []
        multi_particle_gnn_net_list = []
        readout_step_mlp_net_list = []


        # this network is use for prepare_q_input for mb
        prepare_data_net = mydevice.load(PWNet(1,pw4mb_dim,train_dict["pw4mb_nnodes"],train_dict['net_nnodes']))

        if train_dict["multi_particle_net_type"] == 'gnn_identity':

            if train_dict["single_particle_net_type"]  == 'mlp_type':
                print("single particle net type mlp ..........")
                assert False, 'Single particle net type implemention for test is now defunct ...' # 20250807

                single_par_mlp_kwargs = {'input_dim': input_dim, 'output_dim': output_dim, 'nnodes' : train_dict['net_nnodes'],
                                         'init_weights' : train_dict['init_weights'],'p': train_dict["net_dropout"]}
                single_particle_net_list.append(mydevice.load(SingleParticleMLPNet(**single_par_mlp_kwargs)))
                single_particle_net_list.append(mydevice.load(SingleParticleMLPNet(**single_par_mlp_kwargs)))

            elif train_dict["single_particle_net_type"]  == 'transformer_type':
                print("single particle net type transformer ..........")

                # two mb transformer net
                single_par_transformer_kwargs = {'input_dim': input_dim, 'output_dim': train_dict["d_model"],
                          'traj_len': round(train_dict["tau_traj_len"] / train_dict["tau_long"]),
                          'ngrids': train_dict["ngrids"], 'd_model': train_dict["d_model"], 'nhead': train_dict["nhead"],
                          'n_encoder_layers': train_dict["n_encoder_layers"], 'p': train_dict["net_dropout"]}
                single_particle_net_list.append(mydevice.load(SingleParticleTransformerNet(**single_par_transformer_kwargs)))
                single_particle_net_list.append(mydevice.load(SingleParticleTransformerNet(**single_par_transformer_kwargs)))

            else:
                assert (False), 'invalid single particle net type given'

            print("multi particle net type gnn identity..........")
            # two mb gnn net
            # number of gnn layers is zero for identity gnn
            multi_par_gnn_kwargs = {'input_dim': train_dict["d_model"], 'output_dim': train_dict["d_model"], 'n_gnn_layers': 0}
            # two mbnet one for updating p, one for updating q
            multi_particle_gnn_net_list.append(mydevice.load(MultiParticlesGraphNet(**multi_par_gnn_kwargs)))
            multi_particle_gnn_net_list.append(mydevice.load(MultiParticlesGraphNet(**multi_par_gnn_kwargs)))


        elif train_dict["multi_particle_net_type"] ==  'gnn_transformer_type':
            print("single particle net type gnn transformer ..........")

            single_par_transformer_kwargs = {'input_dim': input_dim, 'output_dim': train_dict["d_model"],
                      'traj_len': round(train_dict["tau_traj_len"] / train_dict["tau_long"]),
                      'ngrids': train_dict["ngrids"], 'd_model': train_dict["d_model"], 'nhead': train_dict["nhead"],
                      'n_encoder_layers': train_dict["n_encoder_layers"], 'p': train_dict["net_dropout"]}
            single_particle_net_list.append(mydevice.load(SingleParticleTransformerNet(**single_par_transformer_kwargs)))
            single_particle_net_list.append(mydevice.load(SingleParticleTransformerNet(**single_par_transformer_kwargs)))

            print("multi particle net type gnn ..........")
            multi_par_gnn_kwargs = {'input_dim': train_dict["d_model"],'output_dim': train_dict["d_model"],
                                    'n_gnn_layers' : train_dict["n_gnn_layers"], 'attention' : train_dict["edge_attention"]}
            # two mbnet one for updating p, one for updating q
            multi_particle_gnn_net_list.append(mydevice.load(MultiParticlesGraphNet(**multi_par_gnn_kwargs)))
            multi_particle_gnn_net_list.append(mydevice.load(MultiParticlesGraphNet(**multi_par_gnn_kwargs)))

        else:
            assert (False), 'invalid multi_particle_net_type given'


        if train_dict["readout_step_net_type"] == 'mlp_identity':
            print("readout step net type mlp identity..........")

            readout_step_mlp_kwargs = {'input_dim': train_dict["d_model"], 'output_dim': output_dim,
                                      'nnodes' : train_dict["net_nnodes"],'p' : train_dict["net_dropout"],'readout' : False}
            readout_step_mlp_net_list.append(mydevice.load(ReadoutStepMLPNet(**readout_step_mlp_kwargs)))
            readout_step_mlp_net_list.append(mydevice.load(ReadoutStepMLPNet(**readout_step_mlp_kwargs)))

        elif train_dict["readout_step_net_type"] == 'mlp_type':
            print("readout step net type mlp ..........")

            readout_step_mlp_kwargs = {'input_dim': train_dict["d_model"], 'output_dim': output_dim,
                                      'nnodes' : train_dict["net_nnodes"],'p' : train_dict["net_dropout"]}
            readout_step_mlp_net_list.append(mydevice.load(ReadoutStepMLPNet(**readout_step_mlp_kwargs)))
            readout_step_mlp_net_list.append(mydevice.load(ReadoutStepMLPNet(**readout_step_mlp_kwargs)))

        return prepare_data_net, single_particle_net_list, multi_particle_gnn_net_list, readout_step_mlp_net_list

