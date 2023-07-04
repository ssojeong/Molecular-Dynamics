import torch.nn.functional as F
import torch
import numpy as np

from utils.pbc import pbc
from utils.mydevice import mydevice

class loss:

    def __init__(self,potential_function,eweight,poly_deg,eth=1e9):

        self.potential_function = potential_function

        self.loss_dict = { "total" : [], 
                           "*qrmse" : [], "-qmse"  : [], "-qmae" : [],
                           "*prmse" : [], "-pmse"  : [], "-pmae" : [],
                           "*emae"  : [], "-emse"  : [], "*mmae"  : [],
                           "qshape": [], "pshape": [],
                           "eshape": [], "mshape": [] }
        self.ethrsh = torch.tensor(eth)
        self.ethrsh = mydevice.load(self.ethrsh)
        self.eweight = eweight
        self.poly_deg = poly_deg
    
    # =============================================================
    def eval(self,q_list, p_list, q_label, p_label, q_init, p_init, l_list):

        self.nsamples = q_list.shape[0]

        # # To normalize along nparticle, divide mean of nsamples to nparticle
        qrmse_val = self.q_RMSE_loss(q_list, q_label,l_list) # shape [nsamples] -- [ 2.3, 0.2, 4.3 . .]
        qrmse = torch.sum(qrmse_val) / self.nsamples         # mean over samples

        qmse_val = self.q_MSE_loss(q_list, q_label,l_list)
        qmse = torch.sum(qmse_val) / self.nsamples

        qmae_val = self.q_MAE_loss(q_list, q_label,l_list)
        qmae = torch.sum(qmae_val) / self.nsamples
        
        prmse_val = self.p_RMSE_loss(p_list, p_label)
        prmse = torch.sum(prmse_val) / self.nsamples

        pmse_val = self.p_MSE_loss(p_list, p_label)
        pmse = torch.sum(pmse_val) / self.nsamples

        pmae_val = self.p_MAE_loss(p_list, p_label)
        pmae = torch.sum(pmae_val) / self.nsamples

        emae_val = self.conserve_MAE_eloss(q_list,p_list,q_init,p_init,l_list)
        emae = torch.sum(emae_val) / self.nsamples

        emse_val = self.conserve_MSE_eloss(q_list,p_list,q_init,p_init,l_list)
        emse = torch.sum(emse_val) / self.nsamples
        
        # conploss shape [nsamples]
        mmae_val = self.conserve_MAE_mloss(p_list,p_init) 
        mmae = torch.sum(mmae_val) / self.nsamples
        
        qshape = torch.sum(self.loss_shape_func(qrmse_val))/ self.nsamples
        pshape = torch.sum(self.loss_shape_func(prmse_val))/ self.nsamples

        if emae<self.ethrsh:
            eshape = emae
        else:
            eshape = self.ethrsh
        mshape = mmae

        total = self.total_loss(qshape,pshape,eshape,mshape,self.eweight)
  
        self.loss_dict["total"].append(total.item())      
        self.loss_dict["*qrmse"].append(qrmse.item())      
        self.loss_dict["-qmse"].append(qmse.item())      
        self.loss_dict["-qmae"].append(qmae.item())      
        self.loss_dict["*prmse"].append(prmse.item())      
        self.loss_dict["-pmse"].append(pmse.item())      
        self.loss_dict["-pmae"].append(pmae.item())      
        self.loss_dict["*emae"].append(emae.item())      
        self.loss_dict["-emse"].append(emse.item())      
        self.loss_dict["*mmae"].append(mmae.item())      
  
        return total

    # =============================================================
    def clear(self):
        self.loss_dict = { "total" : [], 
                           "*qrmse" : [], "-qmse"  : [], "-qmae" : [],
                           "*prmse" : [], "-pmse"  : [], "-pmae" : [],
                           "*emae"  : [], "-emse"  : [], "*mmae"  : [],
                           "qshape": [], "pshape": [],
                           "eshape": [], "mshape": [] }
    # =============================================================
    def verbose(self,e,lr,mode):

        #print(e)
        for key,value in self.loss_dict.items():
            if len(value)==0:
                print('empty list: key ',key,' value ',value)
                quit()
            mean = np.mean(value)
            stdd = np.std(value)
            if key=='total' or key=='*qrmse' or key=='*prmse' or key=='*emae' or key=='qshape': 
                print('\n {} {} lr {:.2e}'.format(mode,e,lr),end='')
            print(' {} {:.6e} ({:.4e}) '.format(key,mean,stdd),end='')
        print('\n',flush=True)

    # =============================================================
    def total_loss(self,qshape,pshape,eshape,mshape,ew):

        self.loss_dict["qshape"].append(qshape.item())
        self.loss_dict["pshape"].append(pshape.item())
        self.loss_dict["eshape"].append(eshape.item())
        self.loss_dict["mshape"].append(mshape.item())

        return qshape + pshape + ew*eshape + mshape 

    # =============================================================
    def loss_shape_func(self,x):

        loss  = x
        for d in range(2,self.poly_deg+1):
            xt = 2*d*x
            loss = loss + (xt**d)/d
        return loss

    # =============================================================
    def del_q_adjust(self,q_quantity, q_label, l_list):

        dq = q_quantity - q_label
        # shape [nsamples, nparticle, DIM]
        
        pbc(dq, l_list)
        
        return dq

    # =============================================================
    def q_MSE_loss(self,q_quantity, q_label,l_list):

        nsamples, nparticle, DIM = q_label.shape
        dq = self.del_q_adjust(q_quantity,q_label, l_list) # shape is [nsamples, nparticle, DIM]
        d2 = torch.sum(dq * dq,dim=(2)) # shape is [nsamples, nparticle]
        qloss = torch.sum(d2,dim=1) / nparticle # shape [nsamples]
        return qloss
    # =============================================================
    def q_RMSE_loss(self,q_quantity, q_label,l_list):

        nsamples, nparticle, DIM = q_label.shape
        dq = self.del_q_adjust(q_quantity,q_label, l_list) # shape is [nsamples, nparticle, DIM]
        d2 = torch.sqrt(torch.sum(dq * dq,dim=2)) # shape is [nsamples, nparticle]
        qloss = torch.sum(d2,dim=1) / nparticle # shape [nsamples]
        return qloss
    # =============================================================
    def q_MAE_loss(self,q_quantity, q_label,l_list):

        nsamples, nparticle, DIM = q_label.shape
        dq = self.del_q_adjust(q_quantity,q_label, l_list) # shape is [nsamples, nparticle, DIM]
        d2 = torch.sum(torch.abs(dq),dim=2) # shape is [nsamples, nparticle]
        qloss = torch.sum(d2,dim=1) / nparticle # shape [nsamples]
        return qloss

    # =============================================================
    def p_MSE_loss(self,p_list,p_label):

        nparticles = p_list.shape[1]
        dp = p_list - p_label # shape [nsamples,nparticles,dim]
        dp2 = torch.sum(dp*dp, dim = 2) /nparticles  # shape [nsamples,nparticles]
        return torch.sum(dp2, dim = 1)               # shape [nsamples]
 
    # =============================================================
    def p_RMSE_loss(self,p_list,p_label):

        nparticles = p_list.shape[1]
        dp = p_list - p_label # shape [nsamples,nparticles,dim]
        dp2 = torch.sqrt(torch.sum(dp*dp, dim = 2))  # shape [nsamples,nparticles]
        return torch.sum(dp2, dim = 1) / nparticles  # shape [nsamples]
    
    # =============================================================
    def p_MAE_loss(self,p_list,p_label):

        nparticles = p_list.shape[1]
        dp = p_list - p_label # shape [nsamples,nparticles,dim]
        dp2 = torch.sum(torch.abs(dp), dim = 2) /nparticles  # shape [nsamples,nparticles]
        return torch.sum(dp2, dim = 1)               # shape [nsamples]

    # =============================================================
    def conserve_MAE_eloss(self,q_list,p_list,q_init,p_init,l_list):

        nparticles = p_list.shape[1]
        # shape [nsamples]
        pe_init  = self.potential_function.total_energy(q_init,l_list)
        pe_final = self.potential_function.total_energy(q_list,l_list)
        
        ke_init  = torch.sum(p_init*p_init,dim=(1,2))*0.5
        ke_final = torch.sum(p_list*p_list,dim=(1,2))*0.5
        
        de = (ke_final+pe_final) - (ke_init+pe_init)
        
        return torch.abs(de) / nparticles # shape [nsamples]
   
    # =============================================================
    def conserve_MSE_eloss(self,q_list,p_list,q_init,p_init,l_list):

        nparticles = p_list.shape[1]
        # shape [nsamples]
        pe_init  = self.potential_function.total_energy(q_init,l_list)
        pe_final = self.potential_function.total_energy(q_list,l_list)
        
        ke_init  = torch.sum(p_init*p_init,dim=(1,2))*0.5
        ke_final = torch.sum(p_list*p_list,dim=(1,2))*0.5
        
        de = (ke_final+pe_final) - (ke_init+pe_init)
        
        return de*de / nparticles # shape [nsamples]

    # =============================================================
    def conserve_MAE_mloss(self,p_list,p_init):

        nparticles = p_list.shape[1]
        pinit_sum = torch.sum(p_init,dim=1) # shape [nsamples,dim]
        pfinal_sum = torch.sum(p_list,dim=1) # shape [nsamples,dim]
        
        dp = pinit_sum - pfinal_sum # shape [nsamples,dim]
        
        return torch.abs(dp) / nparticles



