

import torch
import torch.optim as optim
import itertools
import torch.nn as nn

from utils.mydevice                  import mydevice
from hamiltonian.lennard_jones2d     import lennard_jones2d
from ML.trainer.loss                 import loss
from ML.networks.mb_neural_net2       import mb_neural_net as mb_net
from ML.networks.pw_neural_net2       import pw_neural_net as pw_net
from ML.LLUF.pw_mlp        import pw_ff
from ML.LLUF.mb_transformer        import mb_ff
#from ML.LLUF.pw_hf        import pw_hf
#from ML.LLUF.mb_hf        import mb_hf
from ML.LLUF.mbpw         import mbpw
from MD.LLUF_MD             import velocity_verletx
from utils.checkpoint                import checkpoint
from utils.accumulator               import accumulator  # update !!

from optimizers.DecayCosineAnnealingWarmRestarts import DecayCosineAnnealingWarmRestarts

class trainer:

    def __init__(self,train_dict,loss_dict):

        self.train_dict = train_dict
    
        mode = train_dict["nn_mode"]

        if mode == 'ff':
            output_dim = 1
        else:
            print('hf not implemented yet')
            quit()

        #self.force_accumulator = accumulator(velocity_verletx.neval)  # HK

        self.mbpw_obj,self.net_list = self.net_builder(train_dict)
        self.mlvv = velocity_verletx(self.mbpw_obj)

        self.opt = optim.SGD(self.mbpw_obj.parameters(),lr=train_dict["lr"])
        #self.opt = optim.Adam(self.mbpw_obj.parameters(),lr=train_dict["lr"])
        #self.sch = optim.lr_scheduler.StepLR(self.opt,train_dict["sch_step"],train_dict["sch_decay"])
        self.sch = DecayCosineAnnealingWarmRestarts(self.opt,train_dict["sch_step"],train_dict["sch_decay"])
        self.opt_tau = optim.SGD(self.mbpw_obj.tau_parameters(),train_dict["tau_lr"])

        lj = lennard_jones2d()
        self.loss_obj = loss(lj,loss_dict["eweight"],loss_dict["polynomial_degree"])

        self.ckpt = checkpoint(self.net_list,self.opt,self.opt_tau,self.sch)

        self.grad_clip = self.train_dict["grad_clip"]
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
    def reset_grad_clip(self,new_grad_clip):
        print('reseting grad clip ',new_grad_clip)
        self.grad_clip = new_grad_clip
    # ==========================================================
    def one_step(self,q_init,p_init,q_label,p_label,l_init):

      
        for net in self.net_list: net.train()

        self.opt.zero_grad()
        self.opt_tau.zero_grad()

        q_predict,p_predict,l_init = self.mlvv.nsteps(q_init,p_init,l_init,self.train_dict["n_chain"])
    
        loss_val = self.loss_obj.eval(q_predict,p_predict,q_label,p_label,q_init,p_init,l_init)
        loss_val.backward()

        self.mbpw_obj.print_max_grad()
        self.mbpw_obj.grad_clip(self.grad_clip)

        self.opt.step()
        self.opt_tau.step()

    # ==========================================================
    def eval(self,e,q_init,p_init,q_label,p_label,l_init):

        for net in self.net_list: net.eval()
        q_predict,p_predict,l_init = self.mlvv.nsteps(q_init,p_init,l_init,self.train_dict["n_chain"])
        loss_val = self.loss_obj.eval(q_predict,p_predict,q_label,p_label,q_init,p_init,l_init)
        for net in self.net_list: net.train() # set back to train mode
    
    # ==========================================================
    def checkpoint(self,filename):
        self.ckpt.save_checkpoint(filename)
    # ==========================================================

    def verbose(self,e,mode):
        cur_lr = self.opt.param_groups[0]['lr']
        self.loss_obj.verbose(e,cur_lr,mode)
        self.loss_obj.clear()
        self.mbpw_obj.verbose(e,mode)
        for net in self.net_list: net.weight_range()

        # ==== show force ====
        #self.force_accumulator.print(e, mode)  # update !! # HK
        #self.force_accumulator.clear()  # update !! # HK 

    # ==========================================================
    def scheduler_step(self):
        self.sch.step()
    # ==========================================================
    def net_builder(self,train_dict):

        nnet = velocity_verletx.nnet
        pwnet_list = []
        mbnet_list = []
        pw4mb_list = []
        output_dim = 2
        for n in range(nnet):
            pwnet_list.append(mydevice.load(pw_net(   n+2 ,output_dim,train_dict["pwnet_nnodes"])))
            mbnet_list.append(mydevice.load(mb_net(12*n+24,output_dim,train_dict["mbnet_nnodes"])))
        for n in range(nnet//2):
            pw4mb_list.append(mydevice.load(pw_net(2,output_dim,train_dict["pw4mb_nnodes"])))

        ngrids = train_dict["ngrids"]
        b      = train_dict["b"]

        # ==== add accumulator ====  # update !!
        print(self.force_accumulator)
        pw_obj = pw_ff(pwnet_list,nnet, self.force_accumulator)
        mb_obj = mb_ff(mbnet_list,pw4mb_list,ngrids,b,nnet, self.force_accumulator)

        mbpw_obj = mbpw(mb_obj,pw_obj)

        # concatenate all network for checkpoint
        net_list = pwnet_list + mbnet_list + pw4mb_list

        for n in net_list: print(n) # print the network

        return mbpw_obj,net_list


