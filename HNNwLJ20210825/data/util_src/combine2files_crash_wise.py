import sys
import torch

from extract2crashfiles import extract2crashfiles
from extract2crashfiles import save_to


if __name__=='__main__':
    ''' given 2 files, extract the data that has intermediate steps of integration.
        combine them and make trajectories of nsamples.  
         '''
    argv = sys.argv

    infile1 = argv[1]
    infile2 = argv[2]
    outfile = argv[3]

    crash_niter1, crash_nct1, crash_niter2, crash_nct2 = extract2crashfiles(infile1, infile2)

    crash_niter_combine = torch.cat((crash_niter1,crash_niter2),dim=0) 
    crash_nct_combine = torch.cat((crash_nct1,crash_nct2),dim=0) 
    print('crash_niter', crash_niter_combine.shape)
    print('crash_nct', crash_nct_combine.shape)
    print('tot crash_nct', torch.sum(crash_nct_combine))

    save_to(outfile, crash_niter_combine, crash_nct_combine)
