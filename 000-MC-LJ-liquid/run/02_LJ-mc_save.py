# 20200521 Lennard-Jones Liquid on 2dim
#
import matplotlib.pyplot as plt
import numpy as np
from random  import seed
from random  import uniform
import os
import argparse

#############################################
def Compute_Forces(pos,DIM,N,BoxSize, R_adj):
    epsilon = 1.0
    # Compute forces on positions using the Lennard-Jones potential
    # Uses double nested loop which is slow O(N^2) time unsuitable for large systems
    Sij = np.zeros(DIM) # Box scaled units
    Rij = np.zeros(DIM) # Real space units
    
    #Set all variables to zero
    LJ = 0.0  #  Sum of Lennard Jones potential

    # Loop over all pairs of particles
    for i in range(N-1):
        for j in range(i+1,N): #i+1 to N ensures we do not double count
            Sij = pos[i,:]-pos[j,:] # Distance in box scaled units
            for l in range(DIM): # Periodic interactions
                if (np.abs(Sij[l])>0.5):
                    Sij[l] = Sij[l] - np.copysign(1.0,Sij[l]) # If distance is greater
                    #than 0.5  (scaled units) then subtract 0.5 to find periodic 
                    #interaction distance.
            Rij = BoxSize * Sij # Scale the box to the real units in this case reduced LJ units

            #Rsqij = np.dot(Rij,Rij) # Calculate the square of the distance

            R = np.sqrt(np.dot(Rij,Rij))  + R_adj
            Rsqij=R**2.0 

            # Calculate LJ potential inside cutoff
            rm2  = 1.0/(Rsqij) # 1/r^2
            rm6  = rm2**3.0 # 1/r^6
            rm12 = rm6**2.0 # 1/r^12
            LJij = epsilon*(4.0*(rm12-rm6)) # 4[1/r^12 - 1/r^6]  
            LJ += LJij

    return LJ         # return 


########## Main ###############################
DIM     =    2 
N       =    2
MCS     = 32400
R_adj   = 0.00

T       = 0.5  # Tc = 0.522(2) for 2d LJ lquid
rho     = 0.366  # rho_c=0.366(9) for 2d LJ liquid 
#rho     = 0.300  # rho_c=0.366(9) for 2d LJ liquid 
#print("N:",N)
#print("density:",rho)

#beta    = 1./T
BoxSize = np.sqrt(N/rho)
seed(124)

# save file
pos_xy = []
## Monte Carlo  
#for  T in np.linspace(0.4, 0.6, 10):

# assign initial position
pos     = np.random.rand(N,DIM)
pos_old = np.zeros([1,DIM])

LJ_old  = 0
LJ_new  = 0

#inverse temperature
beta    = 1./T

# specific heat calc
TE1sum = 0.0; TE2sum = 0.0; Nsum   = 0.0

# acceptance    calc
ACCsum = 0.0;ACCNsum   = 0.0

for mcs in range(MCS):

    # 1 MCS step
    for _ in range(N):

      # Compute  old LJ potential
      LJ_old = Compute_Forces(pos,DIM,N,BoxSize, R_adj) 

      # randomly choose trial particle
      trial   = np.random.randint(0,N)

      # save old coordinates of trial particle
      pos_old = np.copy(pos[trial])

      # generate new coordinates of trial particle
      #pos[trial]= pos_old +  np.random.rand(1,DIM)*0.125
      #pos[trial]= pos_old +  np.random.rand(1,DIM)*0.500
      pos[trial]= pos_old +  (np.random.rand(1,DIM)-0.5)*0.500
      #pos[trial]= pos_old +  np.random.rand(1,DIM)*0.010

      # pbc
      for i in range(DIM):
        period = np.where(pos[:,i] > 1.0)
        pos[period,i]=pos[period,i]-1.0
        period = np.where(pos[:,i] < 0.0)
        pos[period,i]=pos[period,i]+1.0


      # Compute  new LJ potential
      LJ_new = Compute_Forces(pos,DIM,N,BoxSize, R_adj) 

      # Metropolis algorithm 
      dU = LJ_new - LJ_old
      ACCsum +=1.0; ACCNsum +=1.0
      if( dU > 0 ) :
        if(np.random.rand() > np.exp(-beta*dU) ) :
          ACCsum -= 1.0     #rejected
          pos[trial] = pos_old   
          LJ_new     = LJ_old

    if(mcs >= 400 and mcs%10==0 ): 
      k = BoxSize;
      TE1sum += LJ_new;
      TE2sum += (LJ_new*LJ_new);
      Nsum   += 1.0
      pos_xy.append(np.array(pos))

#print(np.array(pos_xy))
np.save('pos_sampled.npy',np.array(pos_xy))
