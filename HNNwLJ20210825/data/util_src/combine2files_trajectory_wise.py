import sys
import torch

from extract2files import extract2files
from extract2files import save_to


if __name__=='__main__':
    ''' given 2 files, extract the data that has intermediate steps of integration.
        combine them and make trajectories of nsamples.  
         '''
    argv = sys.argv

    infile1 = argv[1]
    infile2 = argv[2]
    outfile = argv[3]

    qp1, qp2, tau_short, tau_long, boxsize = extract2files(infile1, infile2)
    # qp1.shape = [nsamples, (q,p), trajectory, nparticles, DIM]
    # qp2.shape = [nsamples, (q,p), trajectory, nparticles, DIM]

    qp_combine = torch.cat((qp1,qp2),dim=2) # trajectory wise
    print('qp', qp_combine.shape)

    save_to(outfile, qp_combine, tau_short, tau_long, boxsize)
