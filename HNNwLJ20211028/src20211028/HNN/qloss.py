import torch

def del_q_adjust(q_quantity, q_label, phase_space):
    ''' function to adjust difference btw predicted q and q to be no over half of boxsize

    Parameters
    ----------
    q : torch.tensor
            shape is [nsamples, nparticle, DIM]
    boxsize : float

    Returns
    ----------
    '''
    boxsize = phase_space.get_l_list()
    # shape [nsamples, nparticle, DIM]

    dq = q_quantity - q_label
    # shape [nsamples, nparticle, DIM]

    phase_space.adjust_real_q(dq, boxsize)

    return dq

def q_MSE_loss(q_quantity, q_label, phase_space):
    '''
    Parameters
    ----------
    -q_quantity : torch.tensor
    quantities related to q
    -q_label : torch.tensor
    label of related q quantities

    Returns
    ----------
    loss : float
        q MSE loss calculated
    '''

    nsamples, nparticle, DIM = q_label.shape

    dq = del_q_adjust(q_quantity, q_label, phase_space)
    # shape is [nsamples, nparticle, DIM]

    d2 = torch.sum(dq * dq)
    # shape is [nsamples, nparticle, DIM]

    qloss = torch.sum(d2) / nsamples / nparticle

    return qloss

def q_l1_loss(q_quantity, q_label, phase_space):
    '''
    Parameters
    ----------
    -q_quantity : torch.tensor
    quantities related to q
    -q_label : torch.tensor
    label of related q quantities

    Returns
    ----------
    loss : float
        q MAE loss calculated
    '''

    nsamples, nparticle, DIM = q_label.shape

    dq= del_q_adjust(q_quantity, q_label, phase_space)
    # shape is [nsamples, nparticle, DIM]

    d = torch.abs(dq)
    # shape is [nsamples, nparticle, DIM]

    qloss = torch.sum(d) / nsamples / nparticle

    return qloss


