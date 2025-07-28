import torch
import numpy as np

 
# ======================================================
def pbc(q_list,l_list):
    #print('pbc: q list shape ',q_list.shape)
    idx = torch.where(torch.abs(q_list)>0.5*l_list)
    #print('idx ',idx)
    q_list[idx] = q_list[idx] - torch.round(q_list[idx]/l_list[idx])*l_list[idx]
    return q_list
# ======================================================
# calculate the differences in position of two samples with pbc applied
# this code is use to calcualte the qloss

def single_particle_dq_pbc(q_list0,q_list1,l_list):

    dq = q_list0-q_list1
    #print('q list shape ',q_list0.shape,q_list1.shape)
    return pbc(dq,l_list)

# ======================================================
# calculate all pairwise vector of many particles in a box
#
def pairwise_dq_pbc(q_list, l_list):

    dq_list = _delta_state(q_list)

    llist0 = torch.unsqueeze(l_list, dim=1)
    llistm = torch.repeat_interleave(llist0, l_list.shape[1], dim=1)
    # lstatem.shape is [nsamples, nparticle, nparticle, DIM]

    indices = torch.where(torch.abs(dq_list) > 0.5 * llistm)

    dq_list[indices] = dq_list[indices] - torch.round(dq_list[indices] / llistm[indices]) * llistm[indices]

    return dq_list # shape = [nsamples,nparticles,nparticles,dim]
# ======================================================
def _delta_state(state_list):
    state_len = state_list.shape[1]  # nparticle
    state0 = torch.unsqueeze(state_list, dim=1)
    # shape is [nsamples, 1, nparticle, DIM]

    statem = torch.repeat_interleave(state0, state_len, dim=1)
    # shape is [nsamples, nparticle, nparticle, DIM] [[q1, q2, q3, q4],[q1, q2, q3, q4],...,[q1, q2, q3, q4]]

    statet = statem.permute(0,2,1,3)
    # shape is [nsamples, nparticle, nparticle, DIM] [[q1, q1, q1, q1],[q2, q2, q2, q2],...,[q4, q4, q4, q4]]

    dstate = statet - statem
    # shape is [nsamples, nparticle, nparticle, DIM] [[q1-q1, q1-q2, q1-q3, q1-q4],...,[q4-q1, q4-q2, q4-q3, q4-q4]]
    return dstate # shape = [nsamples,nparticles,nparticles,dim]
# ======================================================

def check_pairwise_dq_pbc():
    nsamples = 10  #100
    nparticles = 3
    dim = 2
    eps = 5e-7

    # l_list = 2*torch.rand([nsamples])+0.3
    # l_list= l_list.reshape(nsamples,1,1)
    # l_list = torch.repeat_interleave(l_list,nparticles,dim=1)
    # l_list = torch.repeat_interleave(l_list,dim,dim=2)
    l_list = 2*torch.rand([nsamples,1,dim])+0.3
    l_list = torch.repeat_interleave(l_list,nparticles,dim=1)
    q_list = l_list*(torch.rand([nsamples,nparticles,dim])-0.5)

    dq = pairwise_dq_pbc(q_list,l_list)

    for s in range(nsamples):
        # lstatem.shape is [nsamples, nparticle, nparticle, DIM]
        for i in range(nparticles):
            for j in range(nparticles):

                dqij = dq[s,i,j] # shape [dim]
                minqij = torch.tensor([1000.,1000.]) # shape [dim]
                for dx in range(-1,2):
                    for dy in range(-1,2):

                        qix = q_list[s,i,0]+dx*l_list[s,i,0]
                        qiy = q_list[s,i,1]+dy*l_list[s,i,1]
                        qjx = q_list[s,j,0]
                        qjy = q_list[s,j,1]

                        minqij[0] = torch.min(minqij[0], torch.abs(qix-qjx) )
                        minqij[1] = torch.min(minqij[1], torch.abs(qiy-qjy) )

                if (torch.abs(dqij)-minqij > eps).any() == True:
                    print(dqij,minqij)
                    print('error....')
                    quit()

    print('check delta pbc passed...')

# ======================================================
def check_pbc(): # for one particle
    nsamples = 1
    nparticles = 1
    dim = 2
    R = 5
    eps = 5e-7

    ntest = 100

    lxlist = 2*torch.rand([ntest])+0.3
    lylist = 2*torch.rand([ntest])+0.3
    xlist = lxlist*(torch.rand([ntest])-0.5)
    ylist = lylist*(torch.rand([ntest])-0.5)

    for e in range(ntest):

        if e%(ntest//10)==0: print('checking...',e)
        x = xlist[e]
        y = ylist[e]
        Lx = lxlist[e]
        Ly = lylist[e]
    
        q_label = torch.tensor([[[x,y]]],dtype=torch.float64)
        l_list  = torch.tensor([[[Lx,Ly]]],dtype=torch.float64)
    
        for dx in range(-R,R+1):
            for dy in range(-R,R+1):
                offset = torch.tensor([[[dx*Lx,dy*Ly]]])
                off_pt = q_label + offset
                q_list = q_label + offset
                q_ans = pbc(q_list,l_list)
                assert (q_list.shape==q_ans.shape),'pbc shape wrong '
                diff = torch.mean(torch.abs(q_ans-q_label))
                if (diff>=eps):
                    print('at epoch ',e)
                    print('diff ',diff)
                    print('original x,y ',x,y)
                    print('dx ',dx,'dy',dy,'Lx',Lx,'Ly',Ly)
                    print('off_pt ',off_pt)
                    print('q_ans ',q_ans)
                    print('q_label ',q_label)
                    print('offset ',offset)

                assert (diff<eps),'error in pbc calculations'

    print('check pbc passed...')
 
# ======================================================
def check_single_particle_dq_pbc():

    nsamples = 100  #100
    nparticles = 16
    dim = 2
    eps = 5e-7

    l_list  = 2*torch.rand([nsamples,1,dim])+0.3
    l_list  = torch.repeat_interleave(l_list,nparticles,dim=1)
    #l_size = 4
    #x1 = torch.randn((nsamples, nparticles, dim))
    #l_list = torch.full((x1.shape), l_size)
    #print(l_list)
    q_list0 = l_list*(torch.rand([nsamples,nparticles,dim])-0.5)
    q_list1 = l_list*(torch.rand([nsamples,nparticles,dim])-0.5)
    #print(q_list0, q_list1)
    dq_single = single_particle_dq_pbc(q_list0,q_list1,l_list) # debug here; change name

    delq_vec  = []
    delq_norm = []
    for dx in range(-1,2):
        for dy in range(-1,2):
            shiftx = l_list[:,:,0]*dx
            shifty = l_list[:,:,1]*dy
            shift  = torch.stack((shiftx,shifty),dim=-1)
            qshift = q_list0 + shift
            dq = qshift-q_list1
            delq_vec.append(dq)
            delq_norm.append(torch.norm(dq, dim=-1))

    delq_vec  = torch.stack(delq_vec,dim=0)
    delq_norm = torch.stack(delq_norm,dim=0)
    where     = torch.argmin(delq_norm,dim=0)

    # q_list0.shape = [nsamples,nparticles,dim]
    m1 = np.tile(np.expand_dims(np.arange(0, nsamples), -1), [1, nparticles])
    m2 = np.tile(np.expand_dims(np.arange(0, nparticles), 0), [nsamples,1])

    dq_check = delq_vec[where,m1,m2,:] # delq_vec shape [repeat,batch,nparticle,dim]

    # take diff of two tensors
    err1 = (dq_single - dq_check) # debug here; change name
    err2 = err1*err1

    mse = torch.mean(err2)

    print('err1 ',err1)
    print('err2 ',err2)
    print('mse ',mse)
    assert (mse<eps),'error in single_particle_dq_pbc'
    print('check single_particle_dq_pbd passed...')
    
    
# ======================================================
if __name__=='__main__':

    torch.manual_seed(23841)
    #check_pbc()
    #check_pairwise_dq_pbc()
    check_single_particle_dq_pbc()

