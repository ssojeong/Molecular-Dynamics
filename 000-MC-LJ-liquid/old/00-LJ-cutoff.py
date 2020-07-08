# 20200521 Lennard-Jones Liquid on 2dim
#
import matplotlib.pyplot as plt
import numpy as np

r = np.linspace(0.01,3.0,num=500) # Make a radius vector
epsilon = 1. # Energy minimum
sigma = 1. # Distance to zero crossing point
E_LJ = 4.*epsilon*((sigma/r)**12.-(sigma/r)**6.) # Lennard-Jones potential

plt.figure(figsize=[6,6])  # width and height
plt.ylim(-1.1,2.5); plt.xlim(0.7,3.); plt.grid()
plt.plot(r,E_LJ,'r-',linewidth=1,label=r" $LJ\; pot$") # Red line is unshifted LJ

# The cutoff and shifting value
Rcutoff = 2.5
phicutoff = 4.0/(Rcutoff**12)-4.0/(Rcutoff**6) # Shifts the potential so at the cutoff the potential goes to zero

E_LJ_shift = E_LJ - phicutoff # Subtract the value of the potential at r=2.5
plt.plot(r[:415],E_LJ_shift[:415],'b-',linewidth=1,label=r"$LJ\; pot\; shifted$") # Blue line is shifted






plt.legend()
plt.show()
