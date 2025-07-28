
   
def total_momentum(p_list):

    # p_list shape [nsamples,nparticles,dim]
    p_total = torch.sum(p_list,dim=1)
    return p_total


