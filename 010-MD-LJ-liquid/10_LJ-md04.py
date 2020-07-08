#############################################
# 20200525 Lennard-Jones Liquid on 2dim 
# Molecular Dynamics Simulation
#############################################
import matplotlib.pyplot as plt
import numpy as np
from random  import seed
from random  import uniform
#############################################

def Compute_Forces(pos,force,ene_pot,DIM,N,BoxSize, R_adj):
    epsilon = 1.0
    # Compute forces on positions using the Lennard-Jones potential
    # Uses double nested loop which is slow O(N^2) time unsuitable for large systems
    Sij = np.zeros(DIM) # Box scaled units
    Rij = np.zeros(DIM) # Real space units
    
    #set all variables to zero
    ene_pot = ene_pot*0.0
    force = force*0.0
    virial = 0.0

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

            # Calculate LJ force inside cutoff
            rm2  = 1.0/(Rsqij) # 1/r^2
            rm6  = rm2**3.0 # 1/r^6
            rm12 = rm6**2.0 # 1/r^12
            LJij = epsilon*(4.0*(rm12-rm6)) # 4[1/r^12 - 1/r^6]
            dLJij = epsilon*24.0*rm2*(2.0*rm12-rm6) # 24[2/r^14-1/r^8]

            ene_pot[i] = ene_pot[i] + 0.5*LJij    #Accumulate energy
            ene_pot[j] = ene_pot[j] + 0.5*LJij    #Accumulate energy
            virial = virial - dLJij * Rsqij # Virial is needed to calculate the pressure

            force[i,:] = force[i,:] +  dLJij * Sij   # Accumulate forces
            force[j,:] = force[j,:] -  dLJij * Sij   # Fji=-Fij

    return force, np.sum(ene_pot)/N, -virial/DIM    # return 

def Calculate_Temperature(vel,BoxSize,DIM,N):

    ene_kin = 0.0

    for i in range(N):
        real_vel = BoxSize*vel[i,:]
        ene_kin = ene_kin + 0.5*np.dot(real_vel,real_vel)

    ene_kin_aver = 1.0*ene_kin/N
    temperature = 2.0*ene_kin_aver/DIM

    return ene_kin_aver,temperature
########## Main ###############################
############################################################################
## MD Simulation using MC data
############################################################################
DIM        =    2 
N          =   32 
Nsteps     =  10000
deltat     = 0.0032
TRequested =  0.5   # Reduced temperature
DumpFreq   =  100   # Save the position to file every DumpFreq steps
R_adj      = 0.00
#T          = 0.522  # Tc = 0.522(2) for 2d LJ lquid
rho        = 0.366  # rho_c=0.366(9) for 2d LJ liquid 
#BoxSize    = np.sqrt(N/rho)
BoxSize    = 10.0
volume     = BoxSize**DIM
seed(124)

# initial values for MD
m             = 1.0
v_avg         = 0.0 

#v_std  = 1.0 / np.sqrt(beta*m) 
pos     = np.random.rand(N,DIM)
MassCentre = np.sum(pos,axis=0)/N

for i in range(DIM):
    pos[:,i] = pos[:,i] - MassCentre[i]

ene_kin_aver = np.ones(Nsteps)
ene_pot_aver = np.ones(Nsteps)
virial = np.ones(Nsteps)
T = np.ones(Nsteps)
pressure = np.ones(Nsteps)
ene_pot = np.ones(Nsteps)
#pos = np.load('pos_sampled.npy')

#vel     = np.random.normal(v_avg, v_std, (N,DIM) )
#vel     = vel/BoxSize

vel     = np.random.randn(N,DIM)-0.5
force   = np.random.randn(N,DIM)-0.5

for k in range(Nsteps):

    # Refold positions according to periodic boundary conditions
    for i in range(DIM):
       period = np.where(pos[:,i] > 0.5)
       pos[period,i]=pos[period,i]-1.0
       period = np.where(pos[:,i] < -0.5)
       pos[period,i]=pos[period,i]+1.0

    # velocity verlet
    pos    =  pos +  deltat*vel + 0.5*(deltat**2.0)*force

    ene_kin_aver[k], T[k] = Calculate_Temperature(vel,BoxSize,DIM,N)

    chi = np.sqrt(TRequested/T[k])
    vel = chi*vel + 0.5*deltat*force

    force, ene_pot_aver[k], virial[k] = Compute_Forces(pos,force,ene_pot,DIM,N,BoxSize, R_adj)
    vel    =  vel +  0.5*deltat*force/m

    ene_kin_aver[k], T[k] = Calculate_Temperature(vel,BoxSize,DIM,N)

    pressure[k] = rho*T[k] + virial[k]/volume


plt.figure(figsize=[7,12])
plt.subplot(4, 1, 1)
plt.plot(ene_kin_aver,'k-')
plt.ylabel(r"$E_{K}$", fontsize=20)
plt.subplot(4, 1, 2)
plt.plot(ene_pot_aver,'k-')
plt.ylabel(r"$E_{P}$", fontsize=20)
plt.subplot(4, 1, 3)
plt.plot(T,'k-')
plt.ylabel(r"$T$", fontsize=20)
plt.subplot(4, 1, 4)
plt.plot(pressure,'k-')
plt.ylabel(r"$P$", fontsize=20)
plt.show()
