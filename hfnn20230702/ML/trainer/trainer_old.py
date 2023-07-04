
import torch
import torch.optim as optim

from utils.mydevice                  import mydevice
from hamiltonian.lennard_jones2d     import lennard_jones2d
from ML.trainer.loss                 import loss
from ML.networks.unused.mb_mlp_net import mb_neural_net as mb_net
from ML.networks.pw_mlp_net       import pw_neural_net as pw_net
from ML.LLUF.pw_mlp        import pw_ff
from ML.LLUF.mb_transformer        import mb_ff
#from ML.LLUF.pw_hf        import pw_hf
#from ML.LLUF.mb_hf        import mb_hf
from ML.LLUF.mbpw         import mbpw
from MD.LLUF_MD             import velocity_verletx
from utils.checkpoint                import checkpoint

from optimizers.DecayCosineAnnealingWarmRestarts import DecayCosineAnnealingWarmRestarts

class trainer:

    def __init__(self,train_dict,loss_dict):

        self.train_dict = train_dict

        lj = lennard_jones2d()
        self.mbpw_obj,self.net_list = self.net_builder(train_dict)
        self.mlvv = velocity_verletx(self.mbpw_obj)

        self.opt = optim.SGD(self.mbpw_obj.parameters(),lr=train_dict["maxlr"])
        #self.opt = optim.Adam(self.mbpw_obj.parameters(),lr=train_dict["maxlr"])
        #self.sch = optim.lr_scheduler.StepLR(self.opt,train_dict["sch_step"],train_dict["sch_decay"])
        self.sch = DecayCosineAnnealingWarmRestarts(self.opt,train_dict["sch_step"],
                                                    train_dict["sch_decay"],train_dict["minlr"])
        self.opt_tau = optim.SGD(self.mbpw_obj.tau_parameters(),train_dict["tau_lr"])
        self.tau_list = self.opt_tau.param_groups[0]["params"] 

        self.loss_obj = loss(lj,loss_dict["polynomial_degree"],loss_dict["rthrsh"],loss_dict["e_weight"],loss_dict["reg_weight"]) # remove eweight in loss

        self.ckpt = checkpoint(self.net_list,self.tau_list, self.opt,self.opt_tau,self.sch) # SJ

        self.grad_clip = self.train_dict["grad_clip"]
   # ==========================================================
    def reset_opt_lr(self,new_lr):
        print('reseting learning rate to ',new_lr)
        self.opt.param_groups[0]['lr'] = new_lr
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

        tau_traj_len = train_dict["tau_traj_len"]
        tau_long = train_dict["tau_long"]

        factor = int(tau_traj_len//tau_long)
        pw_input_dim    = dim*factor
        mb_input_dim    = 2*ngrids*dim*factor # each of 6 grids, two forces (fx,fy), two p (px,py)
        pw4mb_input_dim = 2

        nnet = 2

        pwnet_list = []
        mbnet_list = []
        pw4mb_list = []
        output_dim = 2

        # two pwnet, one for updating p, one for updating q
        pwnet_list.append(mydevice.load(pw_net(pw_input_dim,output_dim,train_dict["pwnet_nnodes"],train_dict['h_mode'])))
        pwnet_list.append(mydevice.load(pw_net(pw_input_dim,output_dim,train_dict["pwnet_nnodes"],train_dict['h_mode'])))

        # two mbnet,pw4mbnet, one for updating p, one for updating q
        mbnet_list.append(mydevice.load(mb_net(mb_input_dim,output_dim,train_dict["mbnet_nnodes"],train_dict['h_mode'],train_dict["mbnet_dropout"])))
        mbnet_list.append(mydevice.load(mb_net(mb_input_dim,output_dim,train_dict["mbnet_nnodes"],train_dict['h_mode'], train_dict["mbnet_dropout"])))
        # this network is use for prepare_q_input for mb
        pw4mb_list.append(mydevice.load(pw_net(2,output_dim,train_dict["pw4mb_nnodes"],train_dict['h_mode'])))

        pw_obj = pw_ff(pwnet_list,nnet,train_dict["tau_init"])
        mb_obj = mb_ff(mbnet_list,pw4mb_list,ngrids,b,nnet,train_dict["tau_init"])

        mbpw_obj = mbpw(mb_obj,pw_obj)

        # concatenate all network for checkpoint
        net_list = pwnet_list + mbnet_list + pw4mb_list

        for n in net_list: print(n) # print the network

        return mbpw_obj,net_list


