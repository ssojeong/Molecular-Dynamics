import torch

def check_pbc():
    nsamples = 1
    nparticles = 1
    dim = 2
    R = 5
    eps = 5e-7

    ntest = 1000

    lxlist = 2*torch.rand([ntest])+0.3
    lylist = 2*torch.rand([ntest])+0.3
    xlist = lxlist*(torch.rand([ntest])-0.5)
    ylist = lylist*(torch.rand([ntest])-0.5)

    for e in range(ntest):

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
  
# ======================================================
def pbc(q_list,l_list):
    idx = torch.where(torch.abs(q_list)>0.5*l_list)
    q_list[idx] = q_list[idx] - torch.round(q_list[idx]/l_list[idx])*l_list[idx]
    return q_list
# ======================================================
def delta_pbc(q_list, l_list):

    dq_list = delta_state(q_list)

    llist0 = torch.unsqueeze(l_list, dim=1)
    llistm = torch.repeat_interleave(llist0, l_list.shape[1], dim=1)
    # lstatem.shape is [nsamples, nparticle, nparticle, DIM]

    indices = torch.where(torch.abs(dq_list) > 0.5 * llistm)

    dq_list[indices] = dq_list[indices] - torch.round(dq_list[indices] / llistm[indices]) * llistm[indices]

    return dq_list
# ======================================================
def delta_state(state_list):
    state_len = state_list.shape[1]  # nparticle
    state0 = torch.unsqueeze(state_list, dim=1)
    # shape is [nsamples, 1, nparticle, DIM]

    statem = torch.repeat_interleave(state0, state_len, dim=1)
    # shape is [nsamples, nparticle, nparticle, DIM] [[q1, q2, q3, q4],[q1, q2, q3, q4],...,[q1, q2, q3, q4]]

    statet = statem.permute(0,2,1,3)
    # shape is [nsamples, nparticle, nparticle, DIM] [[q1, q1, q1, q1],[q2, q2, q2, q2],...,[q4, q4, q4, q4]]

    dstate = statet - statem
    # shape is [nsamples, nparticle, nparticle, DIM] [[q1-q1, q1-q2, q1-q3, q1-q4],...,[q4-q1, q4-q2, q4-q3, q4-q4]]
    return dstate
# ======================================================

def check_delta_pbc():
    nsamples = 10  #100
    nparticles = 2
    dim = 2
    eps = 5e-7

    # l_list = 2*torch.rand([nsamples])+0.3
    # l_list= l_list.reshape(nsamples,1,1)
    # l_list = torch.repeat_interleave(l_list,nparticles,dim=1)
    # l_list = torch.repeat_interleave(l_list,dim,dim=2)
    l_list = 2*torch.rand([nsamples,1,dim])+0.3
    l_list = torch.repeat_interleave(l_list,nparticles,dim=1)
    print(l_list)
    q_list = l_list*(torch.rand([nsamples,nparticles,dim])-0.5)

    dq = delta_pbc(q_list,l_list)

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

if __name__=='__main__':

    torch.manual_seed(23841)
    #check_pbc()
    check_delta_pbc()

