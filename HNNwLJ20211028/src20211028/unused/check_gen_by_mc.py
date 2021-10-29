from phase_space.phase_space         import phase_space
from src20211025.integrator.linear_integrator import linear_integrator
from src20211025.integrator.methods import linear_velocity_verlet
from src20211025.hamiltonian.noML_hamiltonian import noML_hamiltonian
from src20211025.utils.check4particle_soft_crash import check4particle_soft_crash
from src20211025.utils.data_io import data_io
from src20211025.utils.show_graph import show_graph

import sys
import math
import torch

def load_anydata(infile1):
    ''' load train or valid or train data'''
    init_qp, _, _, boxsize = data_io.read_trajectory_qp(infile1)
    # init_qp_train.shape = [nsamples, (q, p), 1, nparticle, DIM]

    init_q = torch.squeeze(init_qp[:,0,:,:,:], dim=1)
    init_p = torch.squeeze(init_qp[:,1,:,:,:], dim=1)

    phase_space.set_q(init_q)
    phase_space.set_p(init_p)
    phase_space.set_boxsize(boxsize)

    return init_q.shape, boxsize

if __name__ == '__main__':

    ''' potential energy fluctuations for mc steps that appended to nsamples and 
        histogram of nsamples generated from mc simulation'''

    # run something like this
    # python check_gen_by_mc.py ../data/gen_by_MC/xx/filename temp ../data/gen_by_MC/xx/filename temp

    argv = sys.argv

    if len(argv) != 4:
        print('usage <programe> <infile1> <temp> <rho>')
        quit()

    infile1   = argv[1]
    temp1      = argv[2]
    rho        = argv[3]
    # infile2   = argv[3]
    # temp2      = argv[4]

    phase_space = phase_space()
    hamiltonian_obj = noML_hamiltonian()

    terms = hamiltonian_obj.get_terms()

    # file1 data
    q_file1_shape, boxsize = load_anydata(infile1)  # q_file1_shape is [nsamples, nparticle, DIM]
    u_term1 = terms[0].energy(phase_space)

    # # file2 data
    # q_file2_shape, _ = load_anydata(infile2)  # q_file2_shape is [nsamples, nparticle, DIM]
    # u_term2 = terms[0].energy(phase_space)

    show_graph.u_fluctuation(u_term1, temp1, q_file1_shape[1])
    # show_graph.u_fluctuation(u_term2, temp, q_valid_shape[0], 'valid')
    show_graph.u_distribution4nsamples(u_term1, rho, q_file1_shape[1], boxsize, q_file1_shape[0])
    # show_graph.u_distribution4nsamples(u_term2, temp2, q_file2_shape[1], boxsize, q_file2_shape[0])


