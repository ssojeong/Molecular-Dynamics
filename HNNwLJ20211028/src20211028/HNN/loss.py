import torch
import torch.nn.functional as F
from HNN.qloss import q_MSE_loss
from HNN.qloss import q_l1_loss


def qp_MSE_loss(qp_quantities, label, phase_space, w):
    '''
    Parameters
    ----------
    predicted : tuple of length 2, with elements :
            -q_quantity : torch.tensor
            quantities related to q
            -p_quantity : torch.tensor
            quantities related to p
    label : tuple of length 2 with elements :
        -q_label : torch.tensor
        label of related q quantities
        -p_label : torch.tensor
        label of related p quantities
    w : q, p ratio, default: 0.5

    Returns
    ----------
    loss : float
        Total MSE loss calculated
    '''

    q_quantity, p_quantity = qp_quantities
    q_label, p_label = label

    if q_quantity.shape != q_label.shape or p_quantity.shape != p_label.shape:
        print('loss error shape not match ')
        quit()

    nsamples, nparticle, DIM = q_label.shape

    # # To normalize along nparticle, divide mean of nsamples to nparticle
    qloss = q_MSE_loss(q_quantity, q_label, phase_space)
    ploss = F.mse_loss(p_quantity, p_label, reduction='sum') / nsamples / nparticle

    # # === for checking ===
    # #d2 = (p_quantity-p_label)**2
    # dp = p_quantity - p_label
    # d2 = torch.sum(dp*dp)
    # if torch.abs( torch.sum(d2) / nsamples / nparticle - ploss) > 1e-5:
    #     print('mse_p_loss reduction error ....')
    #     quit()
    #
    # else :
    #     print('mse_p_loss correct')

    loss = 2 * (1 - w) * qloss + 2 * w * ploss

    return loss, qloss, ploss


def qp_l1_loss(qp_quantities, label, phase_space):
    '''
    Returns
    ----------
   w : q and p ratio ; setting 0.5
    qloss, ploss : float
        function to calculate L1 loss and use total MAE loss or total exp loss
    '''
    q_quantity, p_quantity = qp_quantities
    q_label, p_label = label

    if q_quantity.shape != q_label.shape or p_quantity.shape != p_label.shape:
        print('loss error shape not match ')
        quit()

    nsamples, nparticle, DIM = q_label.shape

    # # To normalize along nparticle, divide mean of nsamples to nparticle
    qloss = q_l1_loss(q_quantity, q_label, phase_space)
    ploss = F.l1_loss(p_quantity, p_label, reduction='sum') / nsamples / nparticle

    # # === for checking ===
    # d = torch.abs(p_quantity-p_label)
    # if torch.abs( torch.sum(d) / nsamples / nparticle  - ploss) > 1e-5:
    #     print('mae_p_loss reduction error ....')
    #     quit()
    #
    # else :
    #     print('mae_p_loss correct')

    return qloss, ploss


def qp_MAE_loss(qp_quantities, label, phase_space, w):
    '''
    Returns
    ----------
    w : q and p ratio ; setting 0.5
    loss, qloss, ploss : float
        Total MAE loss calculated
    '''

    qloss, ploss = qp_l1_loss(qp_quantities, label, phase_space)
    loss = 2 * (1 - w) * qloss + 2 * w * ploss

    return loss, qloss, ploss


def qp_exp_loss(qp_quantities, label, phase_space, a):
    '''
    Returns
    ----------
    loss, qloss, ploss : float
        Total exp loss calculated
    '''
    qloss, ploss = qp_l1_loss(qp_quantities, label, phase_space)
    loss = 1 - torch.exp(-a * (qloss + ploss)) + (qloss + ploss)

    return loss, qloss, ploss
