from utils.pbc import delta_pbc
from utils.get_paired_distance_indices import get_paired_distance_indices

import torch
import torch.optim as optim

class lennard_jones2d:


    def __init__(self,epsilon=1.0,sigma=1.0):

        self.epsilon = epsilon
        self.s06      = sigma**6
        self.s12     = sigma**12
        self.eps     = 1e-10


    def total_energy(self,q_list,l_list):

        nsample = q_list.shape[0]
        nparticle = q_list.shape[1]
        dim = q_list.shape[2]

        dq  = delta_pbc(q_list,l_list)
        idx = get_paired_distance_indices.get_indices(dq.shape)
        dr  = get_paired_distance_indices.reduce(dq,idx)
        dr  = dr.view([nsample,nparticle,nparticle-1,dim])

        r = torch.sqrt(torch.sum(dr*dr,dim=-1))
        e_list = self.paired_energy(r)
        e_total = torch.sum(e_list,dim=(1,2))*0.5

        return e_total

    def paired_energy(self,r):

        pair06 = (self.s06  / (r**6  + self.eps))
        pair12 = (self.s12 / (r**12 + self.eps))

        return 4*self.epsilon*(pair12-pair06)



if __name__=='__main__':

    torch.manual_seed(1232)

    torch.set_default_dtype(torch.float64)

    nsample = 20
    nparticle = 2
    dim = 2

    q_list = torch.rand([nsample,nparticle,dim],requires_grad=True)
    l_list = torch.rand([nsample,dim])+nparticle*nparticle
    l_list = torch.unsqueeze(l_list,dim=1)
    l_list = torch.repeat_interleave(l_list,nparticle,dim=1)

    #print('q_list ',q_list)
    #print('l_list ',l_list)

    dq = delta_pbc(q_list,l_list) # shape [nsample,nparticle,nparticle,dim]
    dr = torch.sqrt(torch.sum(dq*dq,dim=-1)) # shape [nsample,nparticle,nparticle]

    #print('dr slow ',dr)

    e_list = []
    for s in range(nsample):
        e = 0.0
        for p1 in range(nparticle):
            for p2 in range(nparticle):
                if p1 != p2:
                    r = dr[s][p1][p2]
                    e6  = 1/(r**6 +1e-10)
                    e12 = 1/(r**12+1e-10)
                    #print('r ',r,' add to ',4*(e12-e6))
                    e += 4*(e12-e6)
        e_list.append(e*0.5)

    e_tensor = torch.tensor(e_list)

    lj = lennard_jones2d()
    e_total = lj.total_energy(q_list,l_list)

    de = e_tensor - e_total
    diff = torch.mean(de*de)
 
    print('diff ',diff)  

    e_target = -0.8
    nepoch = 1000000
    lr = 1e-3
    verb = nepoch//100
    opt = optim.SGD([q_list],lr)
    for e in range(nepoch):
        opt.zero_grad()
        e_total = lj.total_energy(q_list,l_list)
        de = torch.abs(e_total - e_target)
        mean_e = torch.mean(e_total)
        stdd_e = torch.std(e_total,unbiased=False)
        loss = torch.mean(de)
        loss.backward()
        opt.step()
        if e%verb==0: print(e,'loss ',loss.item(),'mean e ',mean_e.item(),'(',stdd_e.item(),')')
        

