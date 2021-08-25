#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def linear_velocity_verlet(hamiltonian, phase_space, tau_cur, boxsize):
    '''
    velocity verlet integrator method

    Parameters
    ----------
    hamiltonian     : can be ML or noML hamiltonian
    phase_space     : real-space-unit, contains q_list, p_list as input for integration
    tau_cur         : float
    boxsize         : float

    Returns
    -------
    state :
    q shape, p shape = [nsamples, nparticle, DIM]
    noML_dHdq1 shape, noML_dHdq2 shape = [nsamples, nparticle, DIM]
    ML_dHdq1 shape, ML_dHdq2 shape = [nsamples, nparticle, DIM]
    '''

    q = phase_space.get_q()
    p = phase_space.get_p()

    tau = tau_cur

    noML_dHdq1, ML_dHdq1 = hamiltonian.dHdq1(phase_space)

    p = p + tau / 2 * (- (noML_dHdq1 + ML_dHdq1))  # dp/dt
    phase_space.set_p(p)

    q = q + tau * p  # dq/dt = dK/dp = p, q is not in dimensionless unit

    phase_space.adjust_real(q, boxsize) # enforce boundary condition - put particle back into box
    phase_space.set_q(q)

    noML_dHdq2, ML_dHdq2 = hamiltonian.dHdq2(phase_space)

    p = p + tau / 2 * (- (noML_dHdq2 + ML_dHdq2))
    phase_space.set_p(p)  # update state after 1 step

    return q, p, noML_dHdq1, noML_dHdq2, ML_dHdq1, ML_dHdq2


# def linear_velocity_verlet_backward(hamiltonian, phase_space, tau_cur, boxsize): # HK delete??
#     '''
#     backward velocity verlet integrator method
#
#     Returns
#     -------
#     state :
#     q shape is [nsamples, nparticle, DIM]
#     p shape is [nsamples, nparticle, DIM]
#     '''
#
#     q = phase_space.get_q()
#     p = phase_space.get_p()
#     MLdHdq1 = phase_space.get_ml_dhdq1()
#     MLdHdq2 = phase_space.get_ml_dhdq2()
#
#     tau = tau_cur
#     noML_dHdq, _ = hamiltonian.dHdq2(phase_space)
#
#     p = p + (tau / 2) * (noML_dHdq+MLdHdq2)  # dp/dt
#     phase_space.set_p(p)
#
#     q = q - tau * p  # dq/dt = dK/dp = p, q is not in dimensionless unit
#
#     phase_space.adjust_real(q, boxsize) # enforce boundary condition - put particle back into box
#     phase_space.set_q(q)
#
#     noML_dHdq, _ = hamiltonian.dHdq1(phase_space)
#     p = p + (tau / 2) * (noML_dHdq+MLdHdq1)  # dp/dt
#
#     phase_space.set_p(p)  # update state after 1 step
#
#     nan_tensor = torch.sqrt(torch.ones(p.shape)-3)
#     phase_space.set_ml_dhdq1(nan_tensor)
#     phase_space.set_ml_dhdq2(nan_tensor)
#
#     return q, p

linear_velocity_verlet.name = 'linear_velocity_verlet'  # add attribute to the function for marker
