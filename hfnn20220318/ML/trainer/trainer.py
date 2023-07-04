
import torch
import torch.optim as optim
import itertools
import torch.nn as nn

from utils.mydevice              import mydevice
from hamiltonian.lennard_jones2d import lennard_jones2d
from ML.trainer.loss             import loss
from ML.networks.mb_neural_net   import mb_neural_net as mb_net
from ML.networks.pw_neural_net   import pw_neural_net as pw_net
from ML.force_functions.pw_ff    import pw_ff
from ML.force_functions.mb_ff    import mb_ff
from ML.force_functions.pw_hf    import pw_hf
from ML.force_functions.mb_hf    import mb_hf
from ML.force_functions.mbpw     import mbpw
from MD.velocity_verlet          import velocity_verlet
from utils.checkpoint            import checkpoint

from optimizers.DecayCosineAnnealingWarmRestarts import DecayCosineAnnealingWarmRestarts

class trainer:

    def __init__(self,train_dict,loss_dict):

        self.train_dict = train_dict

        pwnet_input = 5 # for pw force function
        mbnet_input = 25 # for mb force function
    
        mode = train_dict["nn_mode"]

        if mode == 'hf':
            output_dim = 1
        else:
            output_dim = 2

        pwnet1 = mydevice.load(pw_net(pwnet_input,output_dim,train_dict["nnodes"]))
        pwnet2 = mydevice.load(pw_net(pwnet_input,output_dim,train_dict["nnodes"]))
        mbnet1 = mydevice.load(mb_net(mbnet_input,output_dim,train_dict["nnodes"]))
        mbnet2 = mydevice.load(mb_net(mbnet_input,output_dim,train_dict["nnodes"]))
        self.net_list = [pwnet1,pwnet2,mbnet1,mbnet2]
    
        print('pwnet1',pwnet1)
        print('pwnet2',pwnet2)
        print('mbnet1',mbnet1)
        print('mbnet2',mbnet2)
    
        if mode == 'hf':
            pwforce = pw_hf(pwnet1,pwnet2,train_dict["force_clip"])
            mbforce = mb_hf(mbnet1,mbnet2,train_dict["ngrids"],train_dict["b"],train_dict["force_clip"])
        else:
            pwforce = pw_ff(pwnet1,pwnet2,train_dict["force_clip"])
            mbforce = mb_ff(mbnet1,mbnet2,train_dict["ngrids"],train_dict["b"],train_dict["force_clip"])

    
        self.mbpwff = mbpw(mbforce,pwforce)

        param = itertools.chain(pwnet1.parameters(),pwnet2.parameters(),mbnet1.parameters(),mbnet2.parameters())
        self.opt = optim.SGD(param,lr=train_dict["lr"])
        #sch = optim.lr_scheduler.StepLR(self.opt,train_dict["sch_step"],train_dict["sch_decay"])
        self.sch = DecayCosineAnnealingWarmRestarts(self.opt,train_dict["sch_step"],train_dict["sch_decay"])
        self.opta = optim.SGD(self.mbpwff.parameters(),train_dict["alpha_lr"])

        self.mlvv = velocity_verlet(self.mbpwff)

        lj = lennard_jones2d()
        self.loss_obj = loss(lj,loss_dict["eweight"],loss_dict["polynomial_degree"])

        self.ckpt = checkpoint(self.net_list,self.mbpwff,self.opt,self.opta,self.sch)
    # ==========================================================
    def load_models(self):
        load_file = self.train_dict["loadfile"]
        if load_file is not None:
            self.ckpt.load_checkpoint(load_file) 
    # ==========================================================
    def reset_opt_lr(self,new_lr):
        print('reseting learning rate to ',new_lr)
        self.opt.param_groups[0]['lr'] = new_lr
    # ==========================================================
    def one_step(self,q_init,p_init,q_label,p_label,l_init):

        for net in self.net_list: net.train()

        self.opt.zero_grad()
        self.opta.zero_grad()

        q_predict,p_predict,l_init = self.mlvv.nsteps(q_init,p_init,l_init,self.train_dict["tau_long"],
                                                      self.train_dict["n_chain"])
    
        loss_val = self.loss_obj.eval(q_predict,p_predict,q_label,p_label,q_init,p_init,l_init)
        loss_val.backward()

        for net in self.net_list:
            nn.utils.clip_grad_value_(net.parameters(),clip_value=self.train_dict["grad_clip"])
        nn.utils.clip_grad_value_(self.mbpwff.alpha1,clip_value=self.train_dict["grad_clip"])
        nn.utils.clip_grad_value_(self.mbpwff.alpha2,clip_value=self.train_dict["grad_clip"])

        self.opt.step()
        self.opta.step()

        #for net in self.net_list: net.clamp_weights()
        self.mbpwff.clamp_alpha()

    # ==========================================================
    def eval(self,e,q_init,p_init,q_label,p_label,l_init):

        for net in self.net_list: net.eval()
        q_predict,p_predict,l_init = self.mlvv.nsteps(q_init,p_init,l_init,self.train_dict["tau_long"],
                                                      self.train_dict["n_chain"])
        loss_val = self.loss_obj.eval(q_predict,p_predict,q_label,p_label,q_init,p_init,l_init)
        for net in self.net_list: net.train() # set back to train mode
    
    # ==========================================================
    def checkpoint(self,filename):
        self.ckpt.save_checkpoint(filename)
    # ==========================================================

    def verbose(self,e,mode):
        cur_lr = self.opt.param_groups[0]['lr']
        a = self.mbpwff.a()
        print('{} {:d} lr {:4e} weight for mb net a1 {:4e} a2 {:4e}'.format(mode,e,cur_lr,a[0],a[1]),flush=True)
        self.loss_obj.print(e,cur_lr,mode)
        self.loss_obj.clear()
        for net in self.net_list: net.weight_range()
 
    # ==========================================================
    def scheduler_step(self):
        self.sch.step()
    # ==========================================================


