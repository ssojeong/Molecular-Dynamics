import torch

from utils.mydevice import mydevice

class fbase:

    def __init__(self,net_list,nnet):
        self.net_list = net_list
        self.tau = torch.ones([nnet],requires_grad=True,device = mydevice.get())

    def cat_qp(self,q_input_list,p_input_list):
        qp_list = q_input_list + p_input_list
        qp_cat = torch.cat(qp_list,dim=-1)
        return qp_cat

    def list2netid(self,q_input_list,p_input_list):
        qp_list = q_input_list + p_input_list
        list_len = len(qp_list)
        return list_len-2

    def eval_base(self,q_input_list,p_input_list):
        x = self.cat_qp(q_input_list,p_input_list)
        dq = torch.mean(torch.stack(q_input_list),dim=0)
        #dq = torch.max(torch.stack(q_input_list),dim=0)
        net_id = self.list2netid(q_input_list,p_input_list)
        target_net = self.net_list[net_id]
        return  net_id,self.evalall(target_net,x,dq)

    def verbose(self,e,label):
        tau_list = self.tau.tolist()
        print(e,label,' '.join('{}:{:.2e}'.format(*k) for k in enumerate(tau_list)))




