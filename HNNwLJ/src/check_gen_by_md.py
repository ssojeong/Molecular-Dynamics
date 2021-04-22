from phase_space.phase_space         import phase_space
from hamiltonian.noML_hamiltonian    import noML_hamiltonian
from utils.data_io                   import data_io

import sys
import torch

if __name__ == '__main__':

    argv = sys.argv

    infile1   = argv[1]

    phase_space = phase_space()
    hamiltonian_obj = noML_hamiltonian()

    qp_trajectory, _, _, boxsize = data_io.read_trajectory_qp(infile1)
    # init_qp.shape = [nsamples, (q, p), 2, nparticle, DIM]
    print(qp_trajectory.shape)

    q_trajectory = qp_trajectory[:,0,:,:,:]

    init_q = q_trajectory[:,0,:,:]
    q_strike_append = q_trajectory[:,1,:,:]
    print(init_q.shape, q_strike_append.shape)

    terms = hamiltonian_obj.get_terms()

    # initial state
    phase_space.set_q(init_q)
    phase_space.set_boxsize(boxsize)
    init_u = terms[0].energy(phase_space)

    # strike append state
    phase_space.set_q(q_strike_append)
    strike_append_u = terms[0].energy(phase_space)

    del_e = strike_append_u - init_u
    print(del_e)
