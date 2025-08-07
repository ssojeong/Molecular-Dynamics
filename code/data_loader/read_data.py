import torch
from torchviz import make_dot

from data_loader.data_io import data_io


# ====================================================== 
def read_data(filename,nchain):

    qpl_list,tau_long,tau_short = data_io.read_trajectory_qpl(filename)

    # shape of qp_list is [nsamples, (q, p,l), trajectory length, nparticle, DIM = 2 or 3]
    nsamples,nvariables,traj_length,nparticles,dim = qpl_list.shape

    print('reading data from ',filename)
    print('nsamples ',nsamples)
    print('nvariables ',nvariables)
    print('trajectory length ',traj_length)
    print('nparticles ',nparticles)
    print('dim ',dim)

    label_idx = 1 # n-chain = 1 step
    #label_idx = 2 # n-chain = 4 steps
    #label_idx = 3 # n-chain = 8 steps
    #label_idx = 4 # n-chain =16 steps

    label_nchain = [0,1,4,8,16]

    assert label_nchain[label_idx]==nchain, 'error in nchain and labels'

    qp_list  = qpl_list[:,:,0,:,:]  # start point
    q_list   = qp_list[:,0,:,:].requires_grad_() # q at start
    p_list   = qp_list[:,1,:,:].requires_grad_() # p at start
    l_list   = qpl_list[:,2,0,:,:]
    qp_label = qpl_list[:, :, label_idx, :, :]
    q_label  = qp_label[:,0,:,:]
    p_label  = qp_label[:,1,:,:]

    return q_list, p_list, l_list, q_label, p_label

# ====================================================== 
def ran_shuffle(q_list,p_list,l_list,q_label,p_label):

    nsamples = q_list.shape[0]
    perm_indx = torch.randperm(nsamples)

    q_list  = q_list[perm_indx]
    p_list  = p_list[perm_indx]
    l_list  = l_list[perm_indx]
    q_label = q_label[perm_indx]
    p_label = p_label[perm_indx]

    return q_list,p_list,l_list,q_label,p_label

# ====================================================== 
#def read_data(nsamples,nparticles,dim=2):
#
#    lr  = torch.rand([nsamples,1,dim],requires_grad=False)
#    # make l_list the same shape as q_list,p_list
#    lr = torch.repeat_interleave(lr,nparticles,dim=1)
#
#    qr  = lr*(torch.rand([nsamples,nparticles,dim])-0.5)
#    #pr  = 0.01*torch.randn([nsamples,nparticles,dim])
#    pr  = torch.zeros([nsamples,nparticles,dim])
#    q_list = qr.clone().detach().requires_grad_()
#    p_list = pr.clone().detach().requires_grad_()
#    l_list = lr.clone().detach().requires_grad_()
#
#    dq = 0.1*torch.ones([nsamples,nparticles,dim])
#    dp = 0.1*torch.ones([nsamples,nparticles,dim])
#    qr = qr+dq
#    pr = pr+dp
#    q_label = qr.clone().detach().requires_grad_()
#    p_label = pr.clone().detach().requires_grad_()
#
#    return q_list,p_list,l_list,q_label,p_label
# 
# ====================================================== 
def make_tree(node):
    dot = make_dot(node)
    dot.render('qlist')
    

if __name__=='__main__':

    nsamples = 2
    nparticles = 2
    dim = 2

    q_list,p_list,l_list,q_label,p_label = read_data(nsamples,nparticles,dim)
    
    z = q_list*p_list*q_label*p_label
    make_tree(z)


