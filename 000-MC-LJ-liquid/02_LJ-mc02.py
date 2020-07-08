# 20200521 Lennard-Jones Liquid on 2dim
#
import matplotlib.pyplot as plt
import numpy as np
from random  import seed
from random  import uniform

#############################################
def Compute_Forces(pos,DIM,N):
    epsilon = 1.0
    # Compute forces on positions using the Lennard-Jones potential
    # Uses double nested loop which is slow O(N^2) time unsuitable for large systems
    Rij = np.zeros(DIM) # Real space units
    
    #Set all variables to zero
    LJ = 0.0  #  Sum of Lennard Jones potential

    # Loop over all pairs of particles
    for i in range(N-1):
        for j in range(i+1,N): #i+1 to N ensures we do not double count
            Rij = pos[i,:]-pos[j,:] # Distance in box scaled units
            for l in range(DIM): # Periodic interactions
                if (np.abs(Rij[l])>0.5):
                    Rij[l] = Rij[l] - np.copysign(1.0,Rij[l]) # If distance is greater
                    #than 0.5  (scaled units) then subtract 0.5 to find periodic 
                    #interaction distance.
            
            Rsqij = np.dot(Rij,Rij) # Calculate the square of the distance
            
            # Calculate LJ potential inside cutoff
            rm2  = 1.0/(Rsqij+0.01) # 1/r^2
            rm6  = rm2**3.0 # 1/r^6
            rm12 = rm6**2.0 # 1/r^12
            LJij = epsilon*(4.0*(rm12-rm6)) # 4[1/r^12 - 1/r^6]  
            LJ += LJij

    return LJ         # return 


########## Main ###############################
DIM     =   2 
N       =   2
MCS     = 500
T       = 3.5 
beta    = 1./T
seed(124)

#pos     = np.random.rand(N,DIM)*BoxSize
pos     = np.random.rand(N,DIM)
pos_old = np.zeros([1,DIM])
LJ_old  = 0
LJ_new  = 0

## Monte Carlo  
for mcs in range(MCS):
  for _ in range(N):
    # pbc
    for i in range(DIM):
      period = np.where(pos[:,i] > 1.0)
      pos[period,i]=pos[period,i]-1.0
      period = np.where(pos[:,i] < 0.0)
      pos[period,i]=pos[period,i]+1.0

      #period = np.where(pos[:,i] > 0.5)
      #pos[period,i]=pos[period,i]-1.0
      #period = np.where(pos[:,i] < -0.5)
      #pos[period,i]=pos[period,i]+1.0
    
      # Compute  old LJ potential
      LJ_old = Compute_Forces(pos,DIM,N) 

      # randomly choose trial particle
      trial   = np.random.randint(0,N)

      # save old coordinates of trial particle
      pos_old = np.copy(pos[trial])
 
      # generate new coordinates of trial particle
      #pos[trial]=np.random.rand(1,DIM)*BoxSize
      pos[trial]=np.random.rand(1,DIM)

      # Compute  new LJ potential
      LJ_new = Compute_Forces(pos,DIM,N) 

      # Metropolis algorithm 
      dU = LJ_new - LJ_old
      if( dU > 0 ) :
         if(np.random.rand() > np.exp(-beta*dU) ) :
           pos[trial]=pos_old   
  
  print(pos[0][0], pos[0][1] ,pos[1][0], pos[1][1], mcs, LJ_new)



