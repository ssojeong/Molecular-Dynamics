from ML.LLUF.mbpw import mbpw
import torch.optim as optim
from MD.LLUF_MD import LLUF_MD
from optimizers.DecayCosineAnnealingWarmRestarts import DecayCosineAnnealingWarmRestarts
import torch

class predicter:

    def __init__(self, mbpw_obj):
        self.mbpw_obj = mbpw_obj
        self.mlvv = LLUF_MD(self.mbpw_obj)

        self.mbpw_obj.eval_mode()
    # ==========================================================
    def prepare_input_list(self,q_traj,p_traj,l_init):

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

        return q_input_list,p_input_list,q_cur,p_cur
    # ==========================================================
    def eval(self,q_input_list,p_input_list,q_cur,p_cur,l_init,n_chain):

        qpl_batch = []
        for chain in range(n_chain):
            q_input_list,p_input_list,q_predict,p_predict,l_init = self.mlvv.one_step(q_input_list,p_input_list,q_cur,p_cur,l_init)
            # q_predict [nsamples,nparticles,dim]

            print('window-sliding step ', chain, ' GPU memory % allocated:', round(torch.cuda.memory_allocated(0)/1024**3,2) ,'GB', '\n')
            #print('GPU memory % cached:', round(torch.cuda.memory_cached(0)/1024**3,2) ,'GB' , '\n')
            nxt_qpl = torch.stack((q_predict, p_predict, l_init), dim=1)
            qpl_batch.append(nxt_qpl)

            q_cur = q_predict
            p_cur = p_predict

        #return q_input_list,p_input_list,q_predict,p_predict,l_init
        return q_input_list, p_input_list, qpl_batch
