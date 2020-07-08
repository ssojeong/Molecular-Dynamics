#############################################
# 20200525 Lennard-Jones Liquid on 2dim 
# Molecular Dynamics Simulation
#############################################
import matplotlib.pyplot as plt
import numpy as np
from random  import seed
from random  import uniform
#############################################
def Compute_Forces(pos,force,DIM,N,BoxSize, R_adj):
    epsilon = 1.0
    # Compute forces on positions using the Lennard-Jones potential
    # Uses double nested loop which is slow O(N^2) time unsuitable for large systems
    Sij = np.zeros(DIM) # Box scaled units
    Rij = np.zeros(DIM) # Real space units
    
    #set all variables to zero
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

            # Calculate LJ force     inside cutoff
            rm2  = 1.0/(Rsqij) # 1/r^2
            rm6  = rm2**3.0 # 1/r^6
            rm12 = rm6**2.0 # 1/r^12
            dLJij = epsilon*24.0*rm2*(2.0*rm12-rm6) # 24[2/r^14-1/r^8]

            virial = virial - dLJij * Rsqij # Virial is needed to calculate the pressure

            force[i,:] = force[i,:] +  dLJij * Sij   # Accumulate forces
            force[j,:] = force[j,:] -  dLJij * Sij   # Fji=-Fij

    return force, -virial/DIM    # return 


########## Main ###############################
DIM     =    2 
N       =    2
Nsteps  =   10
R_adj   = 0.00

T       = 0.522  # Tc = 0.522(2) for 2d LJ lquid
rho     = 0.366  # rho_c=0.366(9) for 2d LJ liquid 
BoxSize = np.sqrt(N/rho)
seed(124)

############################################################################
## MD Simulation using MC data
############################################################################
# initial values for MD
beta          = 1./T
m             = 2.0
v_avg         = 0.0 
time_max      = 1
h             = 0.01   # time_interval

# assign initial Velocity 
v_std  = 1.0 / np.sqrt(beta*m) 
pos     = np.random.rand(N,DIM)
vel     = np.random.normal(v_avg, v_std, (N,DIM) )
force   = np.random.randn(N,DIM)-0.5
virial  = 0.0

Contime  = []
ConKE    = []
ConTemp  = []
for time in np.arange(0, time_max, h):
    # Langevin 
    # velocity verlet
    force,virial  = Compute_Forces(pos,force,DIM,N,BoxSize, R_adj)
    vel    =  vel +  0.5*h*force/m
    pos    =  pos +  0.5*h*vel
    force,virial  = Compute_Forces(pos,force,DIM,N,BoxSize, R_adj)
    vel    =  vel +  0.5*h*force/m
    # Langevin 

    # Kinetic Energy
    KE = 0.0
    for i in range(N):
      real_vel = BoxSize * vel[i,:]
      KE      +=  0.5 * np.dot(real_vel,real_vel) 
      Temp     = 2*KE/N*DIM  
      Contime.append(time)
      ConKE.append(KE)
      ConTemp.append(Temp)


plt.plot(Contime,ConKE,label='Kinetic')
plt.plot(Contime,ConTemp,label='T')
plt.legend();plt.grid()
plt.show()
