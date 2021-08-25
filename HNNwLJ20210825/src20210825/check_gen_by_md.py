from phase_space.phase_space         import phase_space
from hamiltonian.noML_hamiltonian    import noML_hamiltonian
from utils.data_io                   import data_io
import matplotlib.pyplot as plt

import sys
import torch
import math

if __name__ == '__main__':

    '''python check_gen_by_md.py ../data/gen_by_MD/train/xxx.pt'''

    argv = sys.argv

    infile1   = argv[1]

    phase_space = phase_space()
    hamiltonian_obj = noML_hamiltonian()

    qp_trajectory, _, _, boxsize = data_io.read_trajectory_qp(infile1)
    # qp_trajectory.shape = [nsamples, (q, p), 2, nparticle, DIM]

    init_qp = qp_trajectory[:,:,0,:,:]
    qp_strike_append = qp_trajectory[:,:,1,:,:]
    # init_qp.shape = [nsamples, (q, p), nparticle, DIM]
    print('initial qp', init_qp.shape)
    print('qp paired with one large time step', qp_strike_append.shape)

    _, _, npar, DIM = init_qp.shape
    terms = hamiltonian_obj.get_terms()

    # initial state
    phase_space.set_q(init_qp[:,0,:,:])
    phase_space.set_p(init_qp[:,1,:,:])
    phase_space.set_boxsize(boxsize)
    #init_u = terms[0].energy(phase_space)

    init_e = hamiltonian_obj.total_energy(phase_space)
    # init_e.shape = [nsamples]

    ########################### thrsh  ##################################
    q_list = init_qp[:,0,:,:] / boxsize
    _, d = phase_space.paired_distance_reduced(q_list, npar, DIM)
    d = d * boxsize
    rthrsh = torch.min(d)

    pothrsh = 4 * (1 / (rthrsh) ** 12 - 1 / (rthrsh) ** 6)
    dhdqthrsh = 4 * ((-12) / (rthrsh) ** 13) - 4 * ((-6) / (rthrsh) ** 7)

    pthrsh = math.sqrt(2 * 1.0) * math.sqrt(-1. * math.log(math.sqrt(2 * math.pi) * 1e-4))
    kethrsh = pthrsh * pthrsh / 2
    ethrsh = kethrsh + pothrsh

    print('rthrsh', rthrsh, 'pothrsh', pothrsh, 'dhdqthrsh', dhdqthrsh, 'pthrsh', pthrsh)

    ###################### check min-max f1 ############################
    dhdq1 = terms[0].evaluate_derivative_q(phase_space)
    prev_f1_sum = torch.sum(torch.square(-dhdq1), dim=-1)
    # prev_f1_sum.shape is [nsamples, nparticle]
    prev_f1_magnitude = torch.sqrt(prev_f1_sum)
    # prev_f1_magnitude.shape is [nsamples, nparticle]
    max_f1 = torch.max(prev_f1_magnitude)
    min_f1 = torch.min(prev_f1_magnitude)

    f1_magnitude, _ = torch.max(prev_f1_magnitude, dim=-1)
    # f1_magnitude.shape is [nsamples]

    # strike append state to calculate f2, e, pot, kin
    phase_space.set_q(qp_strike_append[:,0,:,:])
    phase_space.set_p(qp_strike_append[:,1,:,:])

    ###################### check min-max f2 ############################
    strike_append_dhdq = terms[0].evaluate_derivative_q(phase_space)
    prev_f2_sum = torch.sum(torch.square(-strike_append_dhdq), dim=-1)
    # prev_f2_sum.shape is [nsamples, nparticle]
    prev_f2_magnitude = torch.sqrt(prev_f2_sum)
    # prev_f2_magnitude.shape is [nsamples, nparticle]
    max_f2 = torch.max(prev_f2_magnitude)
    min_f2 = torch.min(prev_f2_magnitude)

    f2_magnitude, _ = torch.max(prev_f2_magnitude, dim=-1)
    # f2_magnitude.shape is [nsamples]

    ###################### check total energy ###########################
    strike_append_e = hamiltonian_obj.total_energy(phase_space)

    del_e = strike_append_e - init_e
    print('del e', del_e)

    ##################### check min-max potential #######################
    strike_append_u = terms[0].energy(phase_space)
    # strike_append_u.shape = [nsamples]
    max_u = torch.max(strike_append_u / npar)
    min_u = torch.min(strike_append_u / npar)

    #################### check min-max momentum #####################
    strike_append_p = qp_strike_append[:, 1, :, :]
    # shape is [nsmaples, nparticle, DIM]

    # check min-max momentum
    # compare w pthrsh
    max_p = torch.max(strike_append_p)
    min_p = torch.min(strike_append_p)

    p_dist = strike_append_p.reshape(-1)
    # shape is [nsmaples*nparticle*DIM]

    ##################### check min-max kinetic #######################
    ke = torch.sum(strike_append_p * strike_append_p / 2, dim=1)  
    # sum along nparticle , ke shape is [nsamples, DIM]
    ke = torch.sum(ke, dim=1)  
    # sum along DIM , ke shape is [nsamples]
    #ke = ke / npar  # to compare kethresh. it is per particle. 
    max_ke = torch.max(ke/ npar)
    min_ke = torch.min(ke/ npar)

    ##################### check min-max energy #######################
    max_e = torch.max(strike_append_e / npar)
    min_e = torch.min(strike_append_e/ npar )

    ########################## print  ##############################
    print('dhdqthrsh', dhdqthrsh, 'f1 max', max_f1.item(), 'min', min_f1.item())
    print('dhdqthrsh', dhdqthrsh, 'f2 max', max_f2.item(), 'min', min_f2.item())
    print('pothrsh', pothrsh, 'u max', max_u.item(), 'min', min_u.item())
    print('pthrsh', pthrsh, 'max_p', max_p.item(), 'min_p', min_p.item())
    print('kethrsh', kethrsh, 'ke max', max_ke.item(), 'min', min_ke.item())
    print('ethrsh', ethrsh, 'e max', max_e.item(), 'min', min_e.item())

    ##################### plot distribution #######################
    plt.title('ethrsh {:.4f} min {:.4f} max {:.4f}'.format(ethrsh, min_e,max_e))
    plt.hist((strike_append_e ).detach().numpy(), bins=100, alpha=.5, label= 'energy' )
    plt.xlabel('energy', fontsize=20)
    plt.ylabel('hist', fontsize=20)
    plt.legend()
    plt.show()

    plt.title('kethrsh {:.4f} min {:.4f}, max {:.4f}'.format(kethrsh, min_ke, max_ke))
    plt.hist(ke.detach().numpy(), bins=100, alpha=.5, label= 'kinetic' )
    plt.xlabel('kinetic', fontsize=20)
    plt.ylabel('hist', fontsize=20)
    plt.legend()
    plt.show()

    plt.title('pthrsh {:.4f} min {:.4f}, max {:.4f}'.format(pthrsh, min_p,max_p))
    plt.hist(p_dist.detach().numpy(), bins=100, alpha=.5, label='p strike append' )
    plt.xlabel('p strik append', fontsize=20)
    plt.ylabel('hist', fontsize=20)
    plt.legend()
    plt.show()

    plt.title('pothrsh {:.4f} min {:.4f}, max {:.4f}'.format(pothrsh, min_u, max_u))
    plt.hist(strike_append_u.detach().numpy(), bins=100, alpha=.5, label= 'potential' )
    plt.xlabel('potential', fontsize=20)
    plt.ylabel('hist', fontsize=20)
    plt.legend()
    plt.show()

    plt.title('fthrsh {:.4f} min {:.4f}, max {:.4f}'.format(-dhdqthrsh, min_f1, max_f1))
    plt.hist(f1_magnitude.detach().numpy(), bins=100, alpha=.5, label= 'f1' )
    plt.xlabel('f1', fontsize=20)
    plt.ylabel('hist', fontsize=20)
    plt.legend()
    plt.show()

    plt.title('fthrsh {:.4f} min {:.4f}, max {:.4f}'.format(-dhdqthrsh, min_f2, max_f2))
    plt.hist(f2_magnitude.detach().numpy(), bins=100, alpha=.5, label= 'f2' )
    plt.xlabel('f2', fontsize=20)
    plt.ylabel('hist', fontsize=20)
    plt.legend()
    plt.show()