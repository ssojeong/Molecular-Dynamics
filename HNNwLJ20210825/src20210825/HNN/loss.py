import torch.nn.functional as F

def qp_MSE_loss(qp_quantities, label, w):

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
    w : weight

    Returns
    ----------
    loss : float
        Total MSE loss calculated
    '''

    q_quantity, p_quantity = qp_quantities

    q_label, p_label = label

    if q_quantity.shape != q_label.shape or p_quantity.shape != p_label.shape:
        print('q pred, label',q_quantity.shape,q_label.shape)
        print('p pred, label',p_quantity.shape, p_label.shape)
        print('error shape not match ')
        quit()

    nsamples, nparticle, DIM = q_label.shape

    # # To normalize along nparticle, divide mean of nsamples to nparticle
    qloss = F.mse_loss(q_quantity, q_label, reduction='sum') / nsamples / nparticle
    ploss = F.mse_loss(p_quantity, p_label, reduction='sum') / nsamples / nparticle

    # # === for checking ===
    # square = (q_quantity-q_label)**2
    # print(torch.sum(square) / nsamples / nparticle / DIM)
    #
    # q_sum = F.mse_loss(q_quantity, q_label, reduction='sum') / nparticle / DIM
    #
    # if torch.abs(q_sum / nsamples - qloss) > 1e-5 :
    #     print('mse_loss reduction error ....')
    #     quit()
    # else :
    #     print('mse_loss correct !!')

    loss =  2 * (1 - w) * qloss + 2 * w * ploss

    return loss,qloss,ploss

