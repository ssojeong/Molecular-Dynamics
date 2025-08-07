import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn

from utils.pbc import pbc
from utils.mydevice import mydevice

from hamiltonian.lennard_jones2d import lennard_jones2d  # hk

class loss:

    #def __init__(self,potential_function,poly_deg,rthrsh,ew,repw,repw2=0.01):
    # hk
    def __init__(self,poly_deg, rthrsh,ew,repw,window_sliding,repw2=0.01):

        # lj is use for calculating regularization on replusive force and conservation of energy
        #self.potential_function = potential_function
        self.potential_function = lennard_jones2d() # hk

        # HK20220426
        self.loss_dict = { "total"  : [], 
                           "*qrmse" : [], "-qmse"  : [], "-qmae" : [],
                           "*prmse" : [], "-pmse"  : [], "-pmae" : [],
                           "*krmse"  : [], "-kmae"  : [],
                           "*urmse"  : [], "-umae"  : [],
                           "*emrse"  : [], "-emae"  : [], "*relurep" : [],
                           "qshape" : [], "pshape" : [],
                           "eshape" : [], "mshape" : [], 
                           "rep": [], "poly"   : [] }

        peth = 4 * (1 / (rthrsh) ** 12)

        self.pethrsh = torch.tensor(peth)
        self.pethrsh = mydevice.load(self.pethrsh)
        self.poly_deg = poly_deg

        self.ew  = ew
        self.repw = repw
        self.repw2 = repw2
        self.window_sliding = window_sliding
        self.m = nn.ReLU()

        print('loss initialized: rthrsh', rthrsh , 'pethrsh {:.3f}'.format(peth), 'e weight', ew, 'reg weight', repw, 'reg weight2', repw2, 'weight len',
                self.window_sliding)
    # =============================================================
    def eval(self,q_list, p_list, q_label, p_label, q_init, p_init, l_list,weight):

        self.nsamples = q_list.shape[0]

        # # To normalize along nparticle, divide mean of nsamples to nparticle
        qrmse_val = self.q_RMSE_loss(q_list, q_label,l_list) # shape [nsamples] -- [ 2.3, 0.2, 4.3 . .]
        #print('qrmse_val', qrmse_val.tolist())
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

        # HK
        kmae_val,umae_val,emae_val, _ = self.conserve_MAE_eloss(q_list,p_list,q_init,p_init,l_list)
        kmae = torch.sum(kmae_val) / self.nsamples
        umae = torch.sum(umae_val) / self.nsamples
        emae = torch.sum(emae_val) / self.nsamples

        krmse_val,urmse_val,ermse_val, eshape_val = self.conserve_RMSE_eloss(q_list,p_list,q_init,p_init,l_list)
        krmse = torch.sum(krmse_val) / self.nsamples
        urmse = torch.sum(urmse_val) / self.nsamples
        ermse = torch.sum(ermse_val) / self.nsamples
        eshape = torch.sum(eshape_val) / self.nsamples

        relurep, rep = self.potential_rep(q_list,l_list)

        # conploss shape [nsamples]
        mmae_val = self.conserve_MAE_mloss(p_list,p_init)
        mmae = torch.sum(mmae_val) / self.nsamples

        # HK20220426
        self.poly_deg = self.calculate_poly(qrmse.item(),prmse.item())

        qshape = torch.sum(self.loss_shape_func(qrmse_val))/ self.nsamples
        pshape = torch.sum(self.loss_shape_func(prmse_val))/ self.nsamples

        #eshape = ermse
        mshape = mmae

        #total,eweight = self.total_loss(qshape,pshape,eshape,mshape,qrmse.item())
        total = self.total_loss(qshape, pshape, eshape, mshape, rep, qrmse.item(),weight)
        # HK20220426
        self.loss_dict["total"].append(total.item())      
        self.loss_dict["*qrmse"].append(qrmse.item())      
        self.loss_dict["-qmse"].append(qmse.item())      
        self.loss_dict["-qmae"].append(qmae.item())      
        self.loss_dict["*prmse"].append(prmse.item())      
        self.loss_dict["-pmse"].append(pmse.item())      
        self.loss_dict["-pmae"].append(pmae.item())
        self.loss_dict["*krmse"].append(krmse.item())
        self.loss_dict["-kmae"].append(kmae.item())
        self.loss_dict["*urmse"].append(urmse.item())
        self.loss_dict["-umae"].append(umae.item())
        self.loss_dict["*ermse"].append(ermse.item())
        self.loss_dict["-emae"].append(emae.item())
        #self.loss_dict["*mmae"].append(mmae.item())
        self.loss_dict["*relurep"].append(relurep.item())
        self.loss_dict["rep"].append(rep.item())
        self.loss_dict["poly"].append(self.poly_deg)
  
        return weight*total

    # =============================================================
    def clear(self):
        # HK20220426
        self.loss_dict = { "total"  : [], 
                           "*qrmse" : [], "-qmse"  : [], "-qmae" : [],
                           "*prmse" : [], "-pmse"  : [], "-pmae" : [],
                           "*krmse"  : [], "-kmae"  : [],
                           "*urmse"  : [], "-umae"  : [],
                           "*ermse"  : [], "-emae"  : [], "*relurep" : [],
                           "qshape" : [], "pshape" : [],
                           "eshape" : [], "mshape" : [],
                            "rep": [], "poly"   : [] }
    # =============================================================
    def verbose(self,e,lr,mode):

        #print(e)
        for key,value in self.loss_dict.items():
            if len(value)==0:
                print('empty list: key ',key,' value ',value)
                # quit()
                continue  # skip the printing if empty list  # HK 20220508

            # 8: n_window_sliding
            # mean list len is 8
            # mean along no of batch each window_sliding ...
            mean_list = np.mean(np.reshape(value,(-1,self.window_sliding)),axis=0)

            if key=='total' or key=='*qrmse' or key=='*prmse' or key=='*urmse' \
                or key=='*krmse' or  key=='*ermse'  or key=='qshape':
                print('\n {} {} lr {:.2e}'.format(mode,e,lr),end='')
            print(' {} '.format(key) + ' '.join( str(item) for item in mean_list),end='')

        print('\n',flush=True)

    # =============================================================
        # HK20220426
    def calculate_poly(self,qrmse_mean_value,prmse_mean_value):
        # rmse_mean_val = max(qrmse_mean_value,prmse_mean_value)
        # if rmse_mean_val>1.0: return max(self.poly_deg,2)
        # if rmse_mean_val>0.6: return max(self.poly_deg,3)
        # if rmse_mean_val>0.2: return max(self.poly_deg,4)
        # return max(self.poly_deg,5)
        return self.poly_deg
    # =============================================================
    def total_loss(self,qshape,pshape,eshape,mshape,rep,qrmse_mean_value,weight):

        self.loss_dict["qshape"].append(qshape.item())
        self.loss_dict["pshape"].append(pshape.item())
        self.loss_dict["eshape"].append(eshape.item())
        #self.loss_dict["mshape"].append(mshape.item())

        #ew = self.calculate_ew(qrmse_mean_value)

        #return qshape + pshape + ew*eshape + mshape, ew
        return qshape + pshape + self.ew * eshape + self.repw * rep

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
        # HK
        ke_init  = torch.sum(p_init*p_init,dim=(1,2))*0.5
        ke_final = torch.sum(p_list*p_list,dim=(1,2))*0.5

        dk = ke_final - ke_init
        du = pe_final - pe_init

        absdk = torch.abs(dk) / nparticles
        absdu = torch.abs(du) / nparticles
        de = torch.abs(dk + du)/nparticles

        eloss = 1 - torch.exp(-de) * (1 + de)

        return absdk,absdu,de, eloss # shape [nsamples]
   
    # =============================================================
    def conserve_RMSE_eloss(self,q_list,p_list,q_init,p_init,l_list):

        nparticles = p_list.shape[1]
        # shape [nsamples]
        pe_init  = self.potential_function.total_energy(q_init,l_list)
        pe_final = self.potential_function.total_energy(q_list,l_list)
        
        ke_init  = torch.sum(p_init*p_init,dim=(1,2))*0.5
        ke_final = torch.sum(p_list*p_list,dim=(1,2))*0.5
        # HK
        dk = (ke_final - ke_init)
        du = (pe_final - pe_init)
        de = dk + du

        dk_sq = torch.sqrt(dk*dk) / nparticles
        du_sq = torch.sqrt(du*du) / nparticles
        de_sq = torch.sqrt(de * de) / nparticles # shape [nsamples]

        eloss = 1 - torch.exp(-de_sq) * (1 + de_sq)
        
        return dk_sq, du_sq, de_sq, eloss # shape [nsamples]

    # =============================================================
    def conserve_MAE_mloss(self,p_list,p_init):

        nparticles = p_list.shape[1]
        pinit_sum = torch.sum(p_init,dim=1) # shape [nsamples,dim]
        pfinal_sum = torch.sum(p_list,dim=1) # shape [nsamples,dim]

        dp = pinit_sum - pfinal_sum # shape [nsamples,dim]

        return torch.abs(dp) / nparticles

    def potential_rep(self, q_list,l_list):

        rep_pe_max  = self.potential_function.repulsive_energy(q_list,l_list)
        relu_pe = self.m(rep_pe_max - self.pethrsh)

        rep = 1 - torch.exp(-self.repw2 * relu_pe) * (1 +self.repw2 * relu_pe)

        return relu_pe, rep
