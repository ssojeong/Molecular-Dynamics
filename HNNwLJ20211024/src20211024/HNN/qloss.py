import torch

def del_q_adjust(q_quantity, q_label, phase_space):

    boxsize = phase_space.get_boxsize()

    dq = q_quantity - q_label
    # shape [nsamples, nparticle, DIM]

    phase_space.adjust_real(dq, boxsize)

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
        Total MSE loss calculated
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
        Total MAE loss calculated
    '''

    nsamples, nparticle, DIM = q_label.shape

    dq= del_q_adjust(q_quantity, q_label, phase_space)
    # shape is [nsamples, nparticle, DIM]

    d = torch.abs(dq)
    # shape is [nsamples, nparticle, DIM]

    qloss = torch.sum(d) / nsamples / nparticle

    return qloss

