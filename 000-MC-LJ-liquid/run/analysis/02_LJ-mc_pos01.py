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
parser = argparse.ArgumentParser('MC simulation')
args = parser.add_argument('--N', type= int, default = 2, help="Num of atom")
args = parser.add_argument('--rho', type= float, default = 0.01, help="density")
args = parser.add_argument('--temp', type= float, default = 0.1, help="temperature")
args = parser.parse_args()

if not os.path.exists('./rho_cv'):
    os.makedirs('./rho_cv')

DIM     =    2 
MCS     = 6400
R_adj   = 0.00

#T       = 0.522  # Tc = 0.522(2) for 2d LJ lquid
#rho     = 0.366  # rho_c=0.366(9) for 2d LJ liquid 
#rho     = 0.300  # rho_c=0.366(9) for 2d LJ liquid 
N   = args.N
rho =args.rho
T =args.temp
#print("N:",N)
#print("density:",rho)

BoxSize = np.sqrt(N/rho)
seed(124)

# save file
text = ''

## Monte Carlo  

# assign initial position
pos     = np.random.rand(N,DIM)
pos_old = np.zeros([1,DIM])

LJ_old  = 0
LJ_new  = 0

#inverse temperature
#T       = 0.05
beta    = 1./T

# specific heat calc
TE1sum = 0.0; TE2sum = 0.0; Nsum   = 0.0

# acceptance    calc
ACCsum = 0.0;ACCNsum   = 0.0

U = []
Acc_rate = []
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
      #pos[trial]= pos_old +  (np.random.rand(1,DIM)-0.5)         #N=2 T = 1
      #pos[trial]= pos_old +  (np.random.rand(1,DIM)-0.5)        #N=2 T = 0.6 
      #pos[trial]= pos_old +  (np.random.rand(1,DIM)-0.5)        #N=2 T = 0.3,0.5
      #pos[trial]= pos_old +  (np.random.rand(1,DIM)-0.5)*0.15  #N=2 T = .2
      #pos[trial]= pos_old +  (np.random.rand(1,DIM)-0.5)*0.1  #N=2 T = .1
      #pos[trial]= pos_old +  (np.random.rand(1,DIM)-0.5)*0.05  #N=2 T = .05

      #pos[trial]= pos_old +  (np.random.rand(1,DIM)-0.5)*0.7   #N=4 T =.6
      #pos[trial]= pos_old +  (np.random.rand(1,DIM)-0.5)*0.3  #N=4 T = .45
      #pos[trial]= pos_old +  (np.random.rand(1,DIM)-0.5)*0.2  #N=4 T = .35
      #pos[trial]= pos_old +  (np.random.rand(1,DIM)-0.5)*0.07  #N=4 T = .25
      #pos[trial]= pos_old +  (np.random.rand(1,DIM)-0.5)*0.03  #N=4 T = .1

      pos[trial]= pos_old +  (np.random.rand(1,DIM)-0.5)*0.9   #N=4 T =.7
      #pos[trial]= pos_old +  (np.random.rand(1,DIM)-0.5)*0.3  #N=4 T = .5
      #pos[trial]= pos_old +  (np.random.rand(1,DIM)-0.5)*0.2  #N=4 T = .35
      #pos[trial]= pos_old +  (np.random.rand(1,DIM)-0.5)*0.05  #N=4 T = .25
      #pos[trial]= pos_old +  (np.random.rand(1,DIM)-0.5)*0.03  #N=6 T = .1

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

    if(mcs> 400 and mcs%4==0 ): 
      U.append(LJ_new)
      # print coordinate (x1,y1) and (x2,y2) and LJ energy
     
      #print(k*pos[0][0], k*pos[0][1] ,k*pos[1][0], k*pos[1][1], mcs, LJ_new)
  
ACCRatio = ACCsum / ACCNsum 
print("ACCRatio:",ACCRatio)
#text = text +  "{0:.3f}".format(T) + ' ' + str(spec) + ' ' + str(ACCRatio)  + '\n'
plt.cla()
plt.xlim(0,BoxSize)
plt.ylim(0,BoxSize)
plt.title(r'$\rho$={}, BoxSize={:.3f}, T={}'.format(rho,BoxSize,T),fontsize=15)
for i in range(N):
    plt.plot(pos[i,0]*BoxSize,pos[i,1]*BoxSize,'o',markersize=15)
plt.show()


#plt.figure(figsize=[12,5])
#plt.subplot(2,1,1)
plt.title('AccRatio={:.3f}'.format(ACCRatio))
plt.plot(U,'k-')
plt.xlabel('mcs',fontsize=20)
plt.ylabel(r'$U_{ij}$',fontsize=20)
plt.show()
