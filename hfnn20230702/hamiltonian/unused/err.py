from utils.pbc import delta_pbc
from utils.get_paired_distance_indices import get_paired_distance_indices

import torch

class lennard_jones2d:


    def __init__(self):

        pass

    def total_energy(self):

        e_list = self.paired_energy()

        return e_list

    def paired_energy(self):

        return e6


if __name__=='__main__':

    torch.manual_seed(1232)

    torch.set_default_dtype(torch.float64)

    nsample = 2
    nparticle = 3
    dim = 2

    q_list = torch.rand([nsample,nparticle,dim],requires_grad=True)
    l_list = torch.rand([nsample,dim])+5.0
    l_list = torch.unsqueeze(l_list,dim=1)
    l_list = torch.repeat_interleave(l_list,nparticle,dim=1)

    print('q_list ',q_list)
    print('l_list ',l_list)

    dq = delta_pbc(q_list,l_list) # shape [nsample,nparticle,nparticle,dim]
    dr = torch.sqrt(torch.sum(dq*dq,dim=-1)) # shape [nsample,nparticle,nparticle]

    print('dr slow ',dr)

    e_list = []
    for s in range(nsample):
        e = 0.0
        for p1 in range(nparticle):
            for p2 in range(nparticle):
                if p1 != p2:
                    r = dr[s][p1][p2]
                    e6  = 1/(r**6 +1e-10)
                    e12 = 1/(r**12+1e-10)
                    print('r ',r,' add to ',4*(e12-e6))
                    e += 4*(e12-e6)
        e_list.append(e*0.5)

    e_tensor = torch.tensor(e_list)

    lj = lennard_jones2d()
    e_total = lj.total_energy()

    print('e slow method ',e_tensor)
    print('e torch method ',e_total)

    de = e_tensor - e_total
    diff = torch.sum(de*de)
 
    print('diff ',diff)  



