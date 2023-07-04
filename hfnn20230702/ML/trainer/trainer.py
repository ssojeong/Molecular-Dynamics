import torch.optim as optim
from utils.mydevice                  import mydevice
from ML.trainer.loss                 import loss
from ML.networks.unused.mb_mlp_net import mb_mlp_net
from ML.networks.pw_mlp_net       import pw_mlp_net
from ML.networks.mb_transformer_net import mb_transformer_net
from ML.networks.mb_graph_net       import mb_transformer4gnn_net
from ML.networks.mb_graph_net       import mb_gnn_net
from ML.LLUF.pw_mlp        import pw_mlp
from ML.LLUF.pw_null        import pw_null
from ML.LLUF.mb_any        import mb_any
from ML.LLUF.mb_both        import mb_both
# from ML.LLUF.mb_transformer        import mb_transformer
from ML.LLUF.mbpw         import mbpw
from MD.LLUF_MD             import LLUF_MD
from utils.checkpoint                import checkpoint

from optimizers.DecayCosineAnnealingWarmRestarts import DecayCosineAnnealingWarmRestarts

class trainer:

    def __init__(self,train_dict,loss_dict):

        self.train_dict = train_dict

        # lj is use for calculating regularization on replusive force and conservation of energy
        self.mbpw_obj,self.net_list = self.net_builder(train_dict) # hk: for lw transformer
        self.mlvv = LLUF_MD(self.mbpw_obj)

        self.opt = optim.SGD(self.mbpw_obj.parameters(),lr=train_dict["maxlr"])
        #self.opt = optim.Adam(self.mbpw_obj.parameters(),lr=train_dict["maxlr"])
        #self.sch = optim.lr_scheduler.StepLR(self.opt,train_dict["sch_step"],train_dict["sch_decay"])
        self.sch = DecayCosineAnnealingWarmRestarts(self.opt,train_dict["sch_step"],
                                                    train_dict["sch_decay"],train_dict["minlr"])
        self.opt_tau = optim.SGD(self.mbpw_obj.tau_parameters(),train_dict["tau_lr"])
        self.tau_list = self.opt_tau.param_groups[0]["params"] 

        self.loss_obj = loss(loss_dict["polynomial_degree"],loss_dict["rthrsh"],loss_dict["e_weight"],loss_dict["reg_weight"]) # remove eweight in loss

        self.ckpt = checkpoint(self.net_list,self.tau_list, self.opt,self.opt_tau,self.sch) # SJ

        self.grad_clip = self.train_dict["grad_clip"]
   # ==========================================================
    def reset_opt_lr(self,new_lr):
        print('reseting opt sch learning rate to ',new_lr)
        self.opt.param_groups[0]['lr'] = new_lr
        self.sch.cosine_sch.get_lr = new_lr  # add
    # ==========================================================
    def try_reset_grad_clip(self,new_grad_clip):
        #if self.mbpw_obj.max_abs_grad('') < 1.5*new_grad_clip: # HK to control to avoid grad too high
        self.grad_clip = new_grad_clip
        print('reseting grad clip ',new_grad_clip)
   # ==========================================================
    def one_step(self,q_traj,p_traj,q_label,p_label,l_init):
      
        for net in self.net_list: net.train()

        self.opt.zero_grad()
        self.opt_tau.zero_grad()

        nsamples,nparticles,_ = q_label.shape

        # need to mask out particle itself interaction at each mb-net and pw-net
        self.mbpw_obj.make_mask(nsamples,nparticles)

        # q_traj,p_traj [traj,nsamples,nparticles,dim]
        q_traj_list = list(q_traj)
        p_traj_list = list(p_traj)

        q_cur = q_traj_list[-1]
        p_cur = p_traj_list[-1]

        prepare_q_input_netid = 0
        q_input_list = []
        p_input_list = []

        # append over trajectory length
        for q,p in zip(q_traj_list,p_traj_list):
            # q,p shape [nsamples,nparticles,dim]
            q_input_list.append(self.mbpw_obj.prepare_q_input(prepare_q_input_netid,q,p,l_init)) # hk: lw take note
            p_input_list.append(self.mbpw_obj.prepare_p_input(q,p,l_init))

        # q_input_list  [(phi0,dq0),(phi1,dq1),(phi2,dq2),...] : tuple inside list
        # phi is function of q at grid point as input for mb-net, dq is input for pw-net
        # p_input_list  [(pi0,dp0),(pi1,dp1),(pi2,dp2),...]
        # pi is momentum at grid point as input for mb-net, dp is input for pw-net
        _,_,q_predict,p_predict,_ = self.mlvv.nsteps(q_input_list,p_input_list,q_cur,p_cur,
                                                     l_init,self.train_dict["n_chain"])
    
        loss_val = self.loss_obj.eval(q_predict,p_predict,q_label,p_label,q_cur,p_cur,l_init)
        loss_val.backward()

        self.mbpw_obj.grad_clip(self.grad_clip)

        self.opt.step()
        self.opt_tau.step()

    # ==========================================================
    def eval(self,q_traj,p_traj,q_label,p_label,l_init):

        self.mbpw_obj.eval_mode()
   
        nsamples,nparticles,_ = l_init.shape

        # need to mask out particle itself interaction at each mb-net and pw-net
        self.mbpw_obj.make_mask(nsamples,nparticles)

        q_traj_list = list(q_traj)
        p_traj_list = list(p_traj)

        q_cur = q_traj_list[-1]
        p_cur = p_traj_list[-1]

        prepare_q_input_netid = 0
        q_input_list = []
        p_input_list = []
        for q,p in zip(q_traj_list,p_traj_list):
            q_input_list.append(self.mbpw_obj.prepare_q_input(prepare_q_input_netid,q,p,l_init))
            p_input_list.append(self.mbpw_obj.prepare_p_input(q,p,l_init))

        # q_input_list [phi0,phi1,phi2,...] ; phi is function of q at grid point
        # p_input_list [pi0,pi1,pi2,...] ; pi is momentum at grid point
        _,_,q_predict,p_predict,_ = self.mlvv.nsteps(q_input_list,p_input_list,q_cur,p_cur,
                                                     l_init,self.train_dict["n_chain"])

        loss_val = self.loss_obj.eval(q_predict,p_predict,q_label,p_label,q_cur,p_cur,l_init)

        self.mbpw_obj.train_mode()
    
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
        cur_lr = self.opt.param_groups[0]['lr']
        self.loss_obj.verbose(e,cur_lr,mode)
        self.loss_obj.clear()
        self.mbpw_obj.verbose(e,mode)
        for net in self.net_list: net.weight_range()
 
    # ==========================================================
    def scheduler_step(self):
        self.sch.step()
    # ==========================================================
    def net_builder(self,train_dict):

        ngrids = train_dict["ngrids"]
        b      = train_dict["b"]
        dim    = 2
        mb_type = train_dict["mb_type"]
        pw_on_off = train_dict["pw_on_off"]

        tau_traj_len = train_dict["tau_traj_len"]
        tau_long = train_dict["tau_long"]

        factor = round(tau_traj_len/tau_long)
        pw_input_dim    = dim*factor
        mb_input_dim    = 2*ngrids*dim*factor # each of 6 grids, func of q (fx,fy), p (px,py)

        nnet = 2

        pwnet_list = []
        mbnet_list = []
        gnnnet_list = []
        pw4mb_list = []

        output_dim = 2

        # two pwnet, one for updating p, one for updating q
        pwnet_list.append(mydevice.load(pw_mlp_net(pw_input_dim,output_dim,train_dict["pwnet_nnodes"],train_dict['h_mode'])))
        pwnet_list.append(mydevice.load(pw_mlp_net(pw_input_dim,output_dim,train_dict["pwnet_nnodes"],train_dict['h_mode'])))

        # this network is use for prepare_q_input for mb
        pw4mb_list.append(mydevice.load(pw_mlp_net(2,output_dim,train_dict["pw4mb_nnodes"],train_dict['h_mode'])))

        if mb_type == 'mlp_type':
            # two mbnet,pw4mbnet, one for updating p, one for updating q
            # hk: change names
            mbnet_list.append(mydevice.load(mb_mlp_net(mb_input_dim,output_dim,train_dict["mbnet_nnodes"],train_dict['h_mode'],train_dict["mbnet_dropout"])))
            mbnet_list.append(mydevice.load(mb_mlp_net(mb_input_dim,output_dim,train_dict["mbnet_nnodes"],train_dict['h_mode'], train_dict["mbnet_dropout"])))
            mb_obj = mb_transformer(mbnet_list,pw4mb_list,ngrids,b,nnet,train_dict["tau_init"])

        elif mb_type == 'transformer_type':     # --- LW
            # two mb transformer net --- LW
            kwargs = {'traj_len': round(train_dict["tau_traj_len"] / train_dict["tau_long"]),
                      'ngrids': train_dict["ngrids"],
                      'd_model': train_dict["d_model"],
                      'nhead': train_dict["nhead"],
                      'n_encoder_layers': train_dict["n_encoder_layers"],
                      'p': train_dict["mbnet_dropout"]}
            mbnet_list.append(mydevice.load(mb_transformer_net(mb_input_dim,output_dim,**kwargs)))
            mbnet_list.append(mydevice.load(mb_transformer_net(mb_input_dim,output_dim,**kwargs)))
            mb_obj = mb_transformer(mbnet_list,pw4mb_list,ngrids,b,nnet,train_dict["tau_init"])         # no change here

        elif mb_type == 'gnn_type': # SJ h20230627
            # two mb gnn net
            n_gnn_layers = train_dict["n_gnn_layers"] # gnn
            kwargs = {'traj_len': round(train_dict["tau_traj_len"] / train_dict["tau_long"]),
                      'ngrids': train_dict["ngrids"],
                      'd_model': train_dict["d_model"],
                      'nhead': train_dict["nhead"],
                      'n_encoder_layers': train_dict["n_encoder_layers"],  # transformer
                      'p': train_dict["mbnet_dropout"],
                      'readout' : 'False'}

            gnnnet_list.append(mydevice.load(mb_gnn_net(output_dim,n_gnn_layers,train_dict["d_model"])))
            gnnnet_list.append(mydevice.load(mb_gnn_net(output_dim,n_gnn_layers,train_dict["d_model"])))
            mbnet_list.append(mydevice.load(mb_transformer4gnn_net(mb_input_dim,output_dim,**kwargs)))
            mbnet_list.append(mydevice.load(mb_transformer4gnn_net(mb_input_dim,output_dim,**kwargs)))
            mb_obj = mb_gnn(mbnet_list,gnnnet_list,pw4mb_list,ngrids,b,nnet,train_dict["tau_init"])         # no change here

        else:
            assert (False), 'invalid mb_ff type given'

        if pw_on_off == "on":
            pw_obj = pw_mlp(pwnet_list, nnet, train_dict["tau_init"])
            mbpw_obj = mbpw(mb_obj,pw_obj)
            # concatenate all network for checkpoint
            net_list = pwnet_list + mbnet_list + pw4mb_list
        else:
            pw_obj = pw_null(pwnet_list, nnet, train_dict["tau_init"])
            mbpw_obj = mbpw(mb_obj, pw_obj)

            if mb_type == 'transformer_type':
                print('transformer type .... net list len 3')
                net_list = pwnet_list + mbnet_list + pw4mb_list

            else:
                print('gnn type .... net list len 4')
                net_list = pwnet_list + gnnnet_list + mbnet_list + pw4mb_list

        for n in net_list: print(n) # print the network

        return mbpw_obj,net_list


