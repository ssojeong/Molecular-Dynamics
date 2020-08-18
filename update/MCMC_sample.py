import Langevin_Machine_Learning.hamiltonian as Hamiltonian
import Langevin_Machine_Learning.Integrator as Integrator
import Langevin_Machine_Learning.utils as confStat
import Langevin_Machine_Learning.phase_space as phase_space
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser('MC simulation')
args = parser.add_argument('--temp', type= float, default = 0.1, help="temperature")
args = parser.add_argument('--dq', type= float, default = 0.1, help="dq")
args = parser.parse_args()

T =args.temp
dq =args.dq
print("T:",T)
print("dq",dq)
energy = Hamiltonian.Hamiltonian() # energy model container
LJ06 = Hamiltonian.LJ_term(epsilon =1, sigma =1, exponent= 6, boxsize=np.sqrt(4/0.2))
LJ12 = Hamiltonian.LJ_term(epsilon =1, sigma =1, exponent= 12, boxsize=np.sqrt(4/0.2))
energy.append(Hamiltonian.Lennard_Jones(LJ06,LJ12, boxsize=np.sqrt(4/0.2)))

configuration = {
    'kB' : 1.0, # put as a constant
    'Temperature' : T,
    'DIM' : 2,
    'm' : 1,
    'particle' : 4,
    'N' : 1,
    'BoxSize': np.sqrt(4/0.2),
    'hamiltonian' : energy,
    }

integration_setting = {
    'iterations' : 12000,
    'DISCARD' : 2000,
    'dq' : dq,
    }

configuration.update(integration_setting) # combine the 2 dictionaries

MSMC_integrator = Integrator.MCMC(**configuration)
q_hist, PE, ACCRatio = MSMC_integrator.integrate()

plt.title('T={}; AccRatio={:.3f}'.format(configuration["Temperature"],ACCRatio),fontsize=15)
plt.plot(PE,'k-')
plt.xlabel('mcs',fontsize=20)
plt.ylabel(r'$U_{ij}$',fontsize=20)
plt.show()

#print('main.py q_hist',q_hist)
print('main.py PE',PE)
base_library = os.path.abspath('Langevin_Machine_Learning/init')
np.save(base_library+ "/N_particle{}_rho{}_T{}_pos_sampled.npy".format(configuration["particle"],0.2,configuration["Temperature"]),q_hist)
