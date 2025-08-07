import time
import torch

class predicter:

    def __init__(self, prepare_data_obj, mlvv):
        self.prepare_data_obj = prepare_data_obj
        self.mlvv = mlvv

        #self.mlvv.eval()
    # ==========================================================
    def prepare_input_list(self,q_traj,p_traj,l_init):

        nsamples,nparticles,_ = l_init.shape

        q_traj_list = list(q_traj)
        p_traj_list = list(p_traj)

        q_cur = q_traj_list[-1]
        p_cur = p_traj_list[-1]

        q_input_list = []
        p_input_list = []
        for q,p in zip(q_traj_list,p_traj_list):
            q_input_list.append(self.prepare_data_obj.prepare_q_feature_input(q,l_init))
            p_input_list.append(self.prepare_data_obj.prepare_p_feature_input(q,p,l_init))

        return q_input_list,p_input_list,q_cur,p_cur
    # ==========================================================
    def eval(self,q_input_list,p_input_list,q_cur,p_cur,l_init,window_sliding, gamma, temp, tau_long):

        start_time = time.time()
        q_input_list,p_input_list,q_predict,p_predict,l_init = self.mlvv.one_step(q_input_list,p_input_list,q_cur,p_cur,l_init, gamma, temp, tau_long)
        # q_predict [nsamples,nparticles,dim]
        end_time = time.time()

        one_step_time = end_time - start_time
        print('nsamples {}'.format(q_cur.shape[0]), 'window_sliding=',window_sliding, '{:.05f} sec'.format(one_step_time))

        #print('window-sliding step ', window_sliding, ' GPU memory % allocated:', round(torch.cuda.memory_allocated(0)/1024**3,2) ,'GB', '\n')
        # print('GPU memory % cached:', round(torch.cuda.memory_cached(0)/1024**3,2) ,'GB' , '\n')

        return q_input_list, p_input_list,q_predict, p_predict, l_init
