#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import copy
import random
import numpy as np
import time
import math
from utils.pbc import pbc
from hamiltonian.lennard_jones2d import lennard_jones2d
from parameters.MC_parameters import MC_parameters
import shutil

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class metropolis_mc:

    ''' This is a Monte Carlo Simulation only used to generate initial positions and sample equilibrium states'''

    _obj_count = 0

    def __init__(self,system_logs):

        metropolis_mc._obj_count += 1
        assert (metropolis_mc._obj_count == 1), type(self).__name__ + " has more than one object"

        self.system_logs = system_logs
        self.boxsize =MC_parameters.boxsize
        self.lennard_jones2d = lennard_jones2d()
        print('metropolis_mc initialized : boxsize ',self.boxsize, flush=True)

    def position_sampler(self):
        if MC_parameters.DIM==2:
            return self.position_sampler2d()
        elif MC_parameters.DIM==3:
            return self.position_sampler3d()
        else:
            assert False, 'dimension of box is limit to 2 or 3  only ..... -.-'


    def position_sampler3d(self):

        ''' function to create random particle positions that are always between -0.5 * boxsize and 0.5 * boxsize

        return : torch.tensor
        shape is [1, nparticle, DIM]
        '''

        # pos = np.random.uniform(-0.5, 0.5, (MC_parameters.nparticle, MC_parameters.DIM))
        # pos = pos * self.boxsize
        # pos = np.expand_dims(pos, axis=0)

        NA = math.ceil((MC_parameters.nparticle)**(1./3)) ** 3
        nasite = math.ceil((NA)**(1./3)) # i < npar condition ensures
        dsite = self.boxsize / nasite
        #print('Nx', nasite, 'Ny', nasite, 'boxsize', self.boxsize, 'dsite', dsite)

        xyz = []
        for nk in range(nasite):
            tmpz = (0.5 + nk) * dsite - self.boxsize / 2
            for ni in range(nasite):
                tmpy = (0.5 + ni) * dsite - self.boxsize / 2
                for nj in range(nasite):
                    tmpx = (0.5 + nj) * dsite - self.boxsize / 2
                    i = nj + ni * nasite + nk * nasite * nasite # pos
                    if i < MC_parameters.nparticle:
                        xyz.append([tmpx, tmpy, tmpz])

        # # visualize
        # xyz = torch.tensor(xyz)
        # print(xyz.shape)
        #
        return torch.unsqueeze(torch.tensor(xyz), dim=0)


    def position_sampler2d(self):

        ''' function to create random particle positions that are always between -0.5 * boxsize and 0.5 * boxsize

        return : torch.tensor
        shape is [1, nparticle, DIM]
        '''

        # pos = np.random.uniform(-0.5, 0.5, (MC_parameters.nparticle, MC_parameters.DIM))
        # pos = pos * self.boxsize
        # pos = np.expand_dims(pos, axis=0)

        NA = math.ceil(math.sqrt(MC_parameters.nparticle)) ** 2
        nasite = int(math.sqrt(NA))
        dsite = self.boxsize / nasite
        #print('Nx', nasite, 'Ny', nasite, 'boxsize', self.boxsize, 'dsite', dsite)

        xy = []
        for ni in range(nasite):
            tmpy = (0.5 + ni) * dsite - self.boxsize / 2
            for nj in range(nasite):
                tmpx = (0.5 + nj) * dsite - self.boxsize / 2
                i = nj + ni * nasite  # pos
                if i < MC_parameters.nparticle:
                    xy.append([tmpx, tmpy])

        return torch.unsqueeze(torch.tensor(xy), dim=0)

    def momentum_dummy_sampler(self):

        ''' function to make momentum zeros because not use for mc simulation

        return : torch.tensor
        shape is [1, nparticle, DIM]
        '''

        momentum = torch.zeros(MC_parameters.nparticle, MC_parameters.DIM)
        momentum = torch.unsqueeze(momentum, dim=0)

        return momentum

    def mcmove(self) :

        ''' MC method. if accepted, move to the new state, but if rejected, remain in the old state.

        parameter
        ------------
        curr_q : shape is [1, npaticle, DIM]
        dq     : float
                At low temperature, mostly reject not update new energy from Boltzmann factor.
                mulitiply displacement to increase acceptance rate

        enn_q  : update potential energy
        eno_q  : old potential energy
        '''
       # shape is [1, npaticle, DIM]

        boxsize = torch.zeros( self.set_q.shape)
        boxsize.fill_(self.set_l)
        # boxsize.shape is [1, npaticle, DIM=2 or 3]

        self.eno_q = self.lennard_jones2d.total_energy(self.set_q, boxsize)

        trial = random.randint(0, self.set_q.shape[1] - 1)           # randomly pick one particle from the state
        old_q = self.set_q[:,trial,:].clone()

        # perform random step with proposed uniform distribution
        # if not move 0.5 , give the particle only a positive displacement
        self.set_q[:, trial] = old_q + (torch.rand(1, MC_parameters.DIM) - 0.5) * self.set_l * MC_parameters.dq

        pbc(self.set_q, boxsize)

        self.enn_q =  self.lennard_jones2d.total_energy(self.set_q, boxsize)

        dU = self.enn_q - self.eno_q

        # accept with probability proportional di e ^ -beta * delta E
        self.ACCsum += 1.0
        self.ACCNsum += 1.0

        if (dU > 0):
            if (torch.rand([]) > math.exp( -dU / MC_parameters.temperature )):
                self.ACCsum -= 1.0      # rejected
                self.set_q[:,trial] = old_q # restore the old position
                self.enn_q = self.eno_q


    def step(self):

        ''' Implementation of integration for Monte Carlo simulation

        parameter
        ___________
        phase space : contains q_list, p_list as input and contains boxsize also
        DISCARD     : discard initial mc steps
        niter       : iteration after discarded mc step
        nsamples    : the number of samples for mc

        Returns
        ___________
        q_list      : shape is [nsample, nparticle, DIM]
        U           : potential energy; shape is [nsamples, niter]
        AccRatio    : acceptance rate to update new energy ; shape is [nsamples]
        spec        : estimate specific heat from fluctuations of the potential energy; shape is [nsamples]

       '''

        niter = MC_parameters.iterations - MC_parameters.DISCARD

        q_list = torch.zeros((MC_parameters.nsamples, niter, MC_parameters.nparticle, MC_parameters.DIM))
        #U = torch.zeros(MC_parameters.nsamples,  MC_parameters.iterations)
        U = torch.zeros(MC_parameters.nsamples, niter)
        ACCRatio = torch.zeros(MC_parameters.nsamples)
        spec = torch.zeros(MC_parameters.nsamples)

        for z in range(0, MC_parameters.nsamples):

            self.ACCsum = 0.
            self.ACCNsum = 0.

            TE1sum = 0.0
            TE2sum = 0.0
            Nsum = 0.0

            self.set_q = self.position_sampler()
            self.set_l = torch.tensor(self.boxsize)

            start = time.time()

            for i in range(0, MC_parameters.iterations):

                for _ in range(MC_parameters.DIM):
                    self.mcmove()

                #U[z, i] = self.enn_q


                if self.enn_q > MC_parameters.nparticle * 10**4:
                    print(f'inital potential energy too high {self.enn_q} ...')
                    file_dir = ''.join(MC_parameters.filename.rsplit('/',maxsplit=1)[:-1])
                    print('file dir', file_dir, ' mv to ', file_dir + '/../tmp/')
                    shutil.move(file_dir, file_dir + '/../tmp/')
                    quit()

                if(i >= MC_parameters.DISCARD):

                    # q_list shape [nsamples, inter, nparticles, dim]
                    q_list[z,i- MC_parameters.DISCARD] = copy.deepcopy(self.set_q)

                    if self.enn_q > MC_parameters.nparticle * MC_parameters.max_energy:

                        print(f'potential energy too high {self.enn_q} ...')
                        quit()

                    U[z,i- MC_parameters.DISCARD] = self.enn_q
                    TE1sum += self.enn_q
                    TE2sum += (self.enn_q * self.enn_q)
                    Nsum += 1.0

                    if i % 100 == 0:
                        print(f's{z} mc_step {i} E {self.enn_q.item()}', flush=True)

            ACCRatio[z] = self.ACCsum / self.ACCNsum
            spec[z] = (TE2sum / Nsum - TE1sum * TE1sum / Nsum / Nsum) / MC_parameters.temperature / MC_parameters.temperature / MC_parameters.nparticle
            #mem = self.system_logs.record_memory_usage('every sample',z + 1)
            end = time.time()

            print('finished taking {} configuration, '.format(z), 'temp: ', MC_parameters.temperature, 'Accratio: ', ACCRatio[z], 'spec: ', spec[z], 'dq: ', MC_parameters.dq, 'rho: ', MC_parameters.rho,  'Discard: ', MC_parameters.DISCARD, 'time: ', end-start, flush=True)

        assert q_list.shape[-1]==MC_parameters.DIM,'wrong q_list shape in metropolis_mc...'

        #print out the rejection rate, recommended rejection 40 - 60 % based on Lit
        return q_list, U, ACCRatio, spec
