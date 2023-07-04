from phase_space.phase_space         import phase_space
from integrator.linear_integrator    import linear_integrator
from integrator.methods              import linear_velocity_verlet
from hamiltonian.noML_hamiltonian    import noML_hamiltonian
from utils.check4particle_crash      import check4particle_crash
from utils.data_io                   import data_io
from utils.show_graph                import show_graph

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

    if len(argv) != 5:
        print('usage <programe> <infile1> <temp> <infile2> <temp>')
        quit()

    infile1   = argv[1]
    temp1      = argv[2]  
    infile2   = argv[3]
    temp2      = argv[4]  

    pthrsh = math.sqrt(2*1.0)*math.sqrt( -1. * math.log(math.sqrt(2*math.pi)*1e-4)) # 4.07
    rthrsh = 0.7

    phase_space = phase_space()
    hamiltonian_obj = noML_hamiltonian()
    crsh_chker = check4particle_crash(rthrsh, pthrsh)
    linear_integrator_obj = linear_integrator(linear_velocity_verlet.linear_velocity_verlet, crsh_chker)

    terms = hamiltonian_obj.get_terms()

    # file1 data
    q_file1_shape, boxsize = load_anydata(infile1)  # q_file1_shape is [nsamples, nparticle, DIM]
    u_term1 = terms[0].energy(phase_space)

    # file2 data
    q_file2_shape, _ = load_anydata(infile2)  # q_file2_shape is [nsamples, nparticle, DIM]
    u_term2 = terms[0].energy(phase_space)

    # show_graph.u_fluctuation(u_term1, temp, q_train_shape[0], 'train')
    # show_graph.u_fluctuation(u_term2, temp, q_valid_shape[0], 'valid')
    show_graph.u_distribution4nsamples(u_term1, temp1, q_file1_shape[1], boxsize, q_file1_shape[0])
    show_graph.u_distribution4nsamples(u_term2, temp2, q_file2_shape[1], boxsize, q_file2_shape[0])


