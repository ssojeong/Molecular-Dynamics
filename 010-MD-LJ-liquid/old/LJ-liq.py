# 20200521 Lennard-Jones Liquid on 2dim
#
import matplotlib.pyplot as plt
import numpy as np

#############################################
def Compute_Forces(pos,acc,ene_pot,epsilon,BoxSize,DIM,N):
    # Compute forces on positions using the Lennard-Jones potential
    # Uses double nested loop which is slow O(N^2) time unsuitable for large systems
    Sij = np.zeros(DIM) # Box scaled units
    Rij = np.zeros(DIM) # Real space units
    
    #Set all variables to zero
    ene_pot = ene_pot*0.0
    acc = acc*0.0

    # The cutoff and shifting value
    Rcutoff=2.5
    phicutoff = 4.0/(Rcutoff**12)-4.0/(Rcutoff**6) # Shifts the potential so 
    # at the cutoff the potential goes to zero

    # Loop over all pairs of particles
    for i in range(N-1):
        for j in range(i+1,N): #i+1 to N ensures we do not double count
            Sij = pos[i,:]-pos[j,:] # Distance in box scaled units
            for l in range(DIM): # Periodic interactions
                if (np.abs(Sij[l])>0.5):
                    Sij[l] = Sij[l] - np.copysign(1.0,Sij[l]) # If distance is greater
                    #than 0.5  (scaled units) then subtract 0.5 to find periodic 
                    #interaction distance.
            
            Rij = BoxSize*Sij # Scale the box to the real units in this case reduced LJ units
            Rsqij = np.dot(Rij,Rij) # Calculate the square of the distance
            
            if(Rsqij < Rcutoff**2):
                # Calculate LJ potential inside cutoff
                # We calculate parts of the LJ potential at a time to improve the 
                #efficieny of the computation (most important for compiled code)
                rm2 = 1.0/Rsqij # 1/r^2
                rm6 = rm2**3.0 # 1/r^6
                rm12 = rm6**2.0 # 1/r^12
                phi = epsilon*(4.0*(rm12-rm6)-phicutoff) # 4[1/r^12 - 1/r^6] - phi(Rc) 
                #- we are using the shifted LJ potential
                # The following is dphi = -(1/r)(dV/dr)
                dphi = epsilon*24.0*rm2*(2.0*rm12-rm6) # 24[2/r^14 - 1/r^8]
                ene_pot[i] = ene_pot[i]+0.5*phi # Accumulate energy
                ene_pot[j] = ene_pot[j]+0.5*phi # Accumulate energy
                acc[i,:] = acc[i,:]+dphi*Sij # Accumulate forces
                acc[j,:] = acc[j,:]-dphi*Sij # (Fji=-Fij)
    #return acc, np.sum(ene_pot)/N      # return 
    return acc, np.sum(ene_pot)         # return 
    #the acceleration vector, potential energy and virial coefficient

########## Main ###############################
DIM     = 2 
N       = 2
BoxSize = 1
Nsteps  = 10
epsilon =1

pos     = np.random.rand(N,DIM)*BoxSize
acc     = np.zeros((N,DIM))
ene_pot = np.zeros(N)
pot_tot  = 0

for k in range(Nsteps):
  # pbc
  for i in range(DIM):
     period = np.where(pos[:,i] > 1.0)
     pos[period,i]=pos[period,i]-1.0
     period = np.where(pos[:,i] < 0.0)
     pos[period,i]=pos[period,i]+1.0
    
  # Compute forces a(t+dt),ene_pot,virial
  acc, pot_tot = Compute_Forces(pos,acc,ene_pot,epsilon,BoxSize,DIM,N) 
  print("%10i %20.10f %20.10f" %(k, acc[0,0]  ,pot_tot))



