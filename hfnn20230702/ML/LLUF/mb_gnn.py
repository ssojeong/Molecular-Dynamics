from ML.LLUF.mb_base import mb_base
import torch
import torch.nn as nn
# ======================================================

class mb_gnn(mb_base):  # SJ h20230627

    def __init__(self,mbnet_list,gnnnet_list,pwnet_list,ngrids,b,nnet,tau_init):

        print('--- initialize mb ff both ---')
        super().__init__(nnet, tau_init)

        self.ngrids = ngrids
        self.b = b

        self.pwnet_list = pwnet_list
        self.gnnnet_list = gnnnet_list
        self.mbnet_list = mbnet_list

        par = []
        for net in gnnnet_list:
            par = par + list(net.parameters())
        for net in mbnet_list:
            par = par + list(net.parameters())
        for net in pwnet_list:
            par = par + list(net.parameters())
        self.param = par
        self.f_stat = force_stat(nnet)

        print('mb both fnn')

    # ===================================================
    def max_abs_grad(self,name): # SJ update
        pw_name = name + '-pw4mb list len' + str(len(self.pwnet_list))
        pw4mb_max_grad = self.get_max_abs_grad(pw_name,self.pwnet_list)

        #20230701 -- need to add max grad for transformer
        # edit here
        mb_name = name + '-mb transformer list len' + str(len(self.mbnet_list))
        mb_transformer_max_grad = self.get_max_abs_grad(mb_name,self.mbnet_list)

        mb_name = name + '-mb gnn list len' + str(len(self.gnnnet_list))
        mb_gnn_max_grad = self.get_max_abs_grad(mb_name,self.gnnnet_list)

        return max(pw4mb_max_grad, mb_transformer_max_grad,mb_gnn_max_grad)
    # ===================================================
    def eval_mode(self):
        self.set_requires_grad_false(self.pwnet_list)
        # 20230701 edit here -- need transformer net
        self.set_requires_grad_false(self.mbnet_list)
        self.set_requires_grad_false(self.gnnnet_list)
    # ===================================================
    def train_mode(self):
        self.set_requires_grad_true(self.pwnet_list)
        # 20230701 edit here -- need transformer net
        self.set_requires_grad_true(self.mbnet_list)
        self.set_requires_grad_true(self.gnnnet_list)
    # ===================================================
    def eval_base(self,net_id,q_input_list,p_input_list, q_pre): # SJ coord
        x = self.cat_qp(q_input_list,p_input_list)
        # mbnet x shape [nsamples * nparticles, ngrids * DIM * (q,p) * traj_len]
        # pwnet x shape [nsamples * nparticles * nparticles, (q,p) * traj_len]
        dq = torch.mean(torch.stack(q_input_list),dim=0)
        input_net = self.mbnet_list[net_id] #20230701 -- SJ
        target_net = self.gnnnet_list[net_id]
        return self.evalall(input_net, target_net,x,dq, q_pre)
    # ===================================================
    def evalall(self,input_net, target_net,x,dq, q_pre): # do not use dq for mb_ff # SJ coord
        nsamples,nparticles,_,_,_ = self.mask.shape
        # x shape [nsamples * nparticles, ngrids * DIM * (q,p) * traj_len]
        # q_pre [nsamples,nparticles,2]
        y = target_net(input_net, x, q_pre) # gnn network
        dim = self.mbnet_list[0].output_dim # check wrong
        # reshape into [nsamples,nparticles,2]
        y3 = y.view([nsamples,nparticles,dim])
        return y3
    # ==================================================
    def eval(self,net_id,q_input_list,p_input_list, q_pre): # SJ coord
        # q_input_list mbnet [phi0, phi1, phi2, ...]
        # p_input_list mbnet [pi0, pi1, pi2, ...]
        y = self.eval_base(net_id,q_input_list,p_input_list, q_pre) #fbase
        f = torch.abs(self.tau[net_id])*( y )
        self.f_stat.accumulate(net_id,y)
        return f
    # ===================================================
    def grad_clip(self,clip_value): # 20230701
        for net in self.gnnnet_list:
            nn.utils.clip_grad_value_(net.parameters(),clip_value)
        for net in self.mbnet_list:
            nn.utils.clip_grad_value_(net.parameters(),clip_value)
        for net in self.pwnet_list:
            nn.utils.clip_grad_value_(net.parameters(),clip_value)
        self.clip_tau_grad(clip_value)


