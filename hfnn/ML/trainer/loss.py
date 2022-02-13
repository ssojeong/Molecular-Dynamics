import torch.nn.functional as F
import torch
import numpy as np

from utils.pbc import pbc
from utils.mydevice import mydevice

class loss:

    def __init__(self,potential_function,eth=1e6):
        self.potential_function = potential_function

        self.loss_dict = { "total" : [], 
                           "*qrmse" : [], "qmse"  : [], "qmae" : [],
                           "*prmse" : [], "pmse"  : [], "pmae" : [],
                           "emae"  : [], "*emse"  : [], "*mmae"  : [],
                           "qshape": [], "pshape": [],
                           "eshape": [], "mshape": [] }
        self.ethrsh = torch.tensor(eth)
        self.ethrsh = mydevice.load(self.ethrsh)
    
    # =============================================================
    def eval(self,q_list, p_list, q_label, p_label, q_init, p_init, l_list, ew=1e-2):

        self.nsamples = q_list.shape[0]

        # # To normalize along nparticle, divide mean of nsamples to nparticle
        qrmse = self.q_RMSE_loss(q_list, q_label,l_list)
        qrmse = torch.sum(qrmse) / self.nsamples
        qmse = self.q_MSE_loss(q_list, q_label,l_list)
        qmse = torch.sum(qmse) / self.nsamples
        qmae = self.q_MAE_loss(q_list, q_label,l_list)
        qmae = torch.sum(qmae) / self.nsamples
        
        prmse = self.p_RMSE_loss(p_list, p_label)
        prmse = torch.sum(prmse) / self.nsamples
        pmse = self.p_MSE_loss(p_list, p_label)
        pmse = torch.sum(pmse) / self.nsamples
        pmae = self.p_MAE_loss(p_list, p_label)
        pmae = torch.sum(pmae) / self.nsamples

        # eloss shape [nsamples]
        emae = self.conserve_MAE_eloss(q_list,p_list,q_init,p_init,l_list)
        emae = torch.sum(emae) / self.nsamples
        emse = self.conserve_MSE_eloss(q_list,p_list,q_init,p_init,l_list)
        emse = torch.sum(emse) / self.nsamples
        
        # conploss shape [nsamples]
        mmae = self.conserve_RMSE_mloss(p_list,p_init) 
        mmae = torch.sum(mmae) / self.nsamples
        
        total = self.total_loss(qrmse,prmse,emse,mmae,ew)
  
        self.loss_dict["total"].append(total.item())      
        self.loss_dict["*qrmse"].append(qrmse.item())      
        self.loss_dict["qmse"].append(qmse.item())      
        self.loss_dict["qmae"].append(qmae.item())      
        self.loss_dict["*prmse"].append(prmse.item())      
        self.loss_dict["pmse"].append(pmse.item())      
        self.loss_dict["pmae"].append(pmae.item())      
        self.loss_dict["emae"].append(emae.item())      
        self.loss_dict["*emse"].append(emse.item())      
        self.loss_dict["*mmae"].append(mmae.item())      
  
        return total

    # =============================================================
    def clear(self):
        self.loss_dict = { "total" : [], 
                           "*qrmse" : [], "qmse"  : [], "qmae" : [],
                           "*prmse" : [], "pmse"  : [], "pmae" : [],
                           "emae"  : [], "*emse"  : [], "*mmae"  : [],
                           "qshape": [], "pshape": [],
                           "eshape": [], "mshape": [] }
    # =============================================================
    def print(self,e,mode):

        #print(e)
        for key,value in self.loss_dict.items():
            if len(value)==0:
                print('empty list: key ',key,' value ',value)
                quit()
            mean = np.mean(value)
            stdd = np.std(value)
            if key=='*qrmse' or key=='*prmse' or key=='emae' or key=='qshape': print('\n',mode,e,end='')
            print(' {} {:.6e} ({:.4e}) '.format(key,mean,stdd),end='')
        print('\n')

    # =============================================================
    def total_loss(self,qloss,ploss,eloss,mloss,ew=1e-2):

        qshape = self.loss_shape_func(qloss)
        pshape = self.loss_shape_func(ploss)
        if eloss<self.ethrsh:
            eshape = eloss
        else:
            eshape = self.ethrsh
        mshape = self.loss_shape_func(mloss)

        self.loss_dict["qshape"].append(qshape.item())
        self.loss_dict["pshape"].append(pshape.item())
        self.loss_dict["eshape"].append(eshape.item())
        self.loss_dict["mshape"].append(mshape.item())

        return qshape + pshape + ew*eshape + mshape 
        # eshape tend to have much greater gradient, so make grad ew smaller

    # =============================================================
    def loss_shape_func(self,x):

        x1 = x
        x2 = 2*x
        x3 = 3*x
        x4 = 4*x
        
        return x1 + x2**2/2 + x3**3/3 + x4**4/4

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
    def conserve_RMSE_mloss(self,p_list,p_init):

        nparticles = p_list.shape[1]
        pinit_sum = torch.sum(p_init,dim=1) # shape [nsamples,dim]
        pfinal_sum = torch.sum(p_list,dim=1) # shape [nsamples,dim]
        
        dp = pinit_sum - pfinal_sum # shape [nsamples,dim]
        dp2 = torch.sum(dp*dp,dim=1) # shape [nsamples]
        
        return torch.sqrt(dp2) / nparticles



