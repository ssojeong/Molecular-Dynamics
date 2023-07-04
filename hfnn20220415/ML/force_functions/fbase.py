import torch

from utils.mydevice import mydevice

class fbase:

    def __init__(self,net_list,neval=6):
        self.net_list = net_list
        self.tau = torch.ones([neval],requires_grad=True,device = mydevice.get())

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
        net_id = self.list2netid(q_input_list,p_input_list)
        target_net = self.net_list[net_id]
        return  net_id,self.evalall(target_net,x)

    def verbose(self,e,label):
        print(e,label,' ',self.tau)




