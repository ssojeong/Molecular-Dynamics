#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

class kinetic_energy:
    '''Implemented class for total kinetic energy of p^2/2m'''
    def __init__(self, mass : float):
        '''
        Parameters
        ----------
        mass : float
            mass of the particles, taken to be constant for all
        '''
        try : 
            str_mass = str(float(mass))
        except : 
            raise Exception('Mass is not a float / error in mass')
            
        self.mass = mass
        print('kinetic_energy.py call kinetic')
        self._name = 'Kinetic Energy'

    def energy(self, phase_space): # phase space is real-space-unit
        p = phase_space.get_p()
        e_element = torch.sum( p*p / (2 * self.mass), dim = 1) # x, y
        e = torch.sum( e_element, dim = 1) # sum each element x, y

        return  e # make zero for Monte-carlo ( torch.zeros([]) )

    def evaluate_derivative_q(self, phase_space):
        return torch.zeros(phase_space.get_q().shape)

    def evaluate_second_derivative_q(self, phase_space):
        nsamples, nparticle, DIM  = phase_space.get_q().shape
        return torch.zeros((nsamples, DIM * nparticle, DIM * nparticle))
