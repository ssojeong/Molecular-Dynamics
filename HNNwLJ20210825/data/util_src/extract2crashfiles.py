import torch

def extract2crashfiles(infile1, infile2):

    # given 2 files, extract the data 
    # are consistent, return
    # crash_niter, crash_nct
 
    data1 = torch.load(infile1)
    data2 = torch.load(infile2)

    crash_niter1 = data1['crash_niter']
    crash_nct1  = data1['crash_nct']
    crash_niter2 = data2['crash_niter']
    crash_nct2  = data2['crash_nct']

    return crash_niter1, crash_nct1, crash_niter2, crash_nct2

def save_to(outfile, crash_niter, crash_nct):
    ''' save multiple components, organize them in a dictionary '''

    data_combine = {'crash_niter' : crash_niter, 'crash_nct' : crash_nct}
    torch.save(data_combine, outfile)

