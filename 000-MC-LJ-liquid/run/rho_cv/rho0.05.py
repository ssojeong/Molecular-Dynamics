import numpy as np
import matplotlib.pyplot as plt

L2 = np.genfromtxt('N2_rho0.05')
L4 = np.genfromtxt('N4_rho0.05')
L6 = np.genfromtxt('N6_rho0.05')

#x = range(0.0500)
T2 = L2[:,0]; cv2 = L2[:,1]; acc_rate2 = L2[:,2]
T4 = L4[:,0]; cv4 = L4[:,1]; acc_rate4 = L4[:,2]
T6 = L6[:,0]; cv6 = L6[:,1]; acc_rate6 = L6[:,2]

#plt.figure(figsize=[10.05])
plt.plot(T2,cv2,label='L=2')
plt.plot(T4,cv4,label='L=4')
plt.plot(T6,cv6,label='L=6')
plt.xlabel('T',fontsize=20)
plt.ylabel(r'$c_{v}$',fontsize=20)
plt.legend(loc='upper right',fontsize=20)
plt.title(r'Fixed $\rho$=0.05',fontsize=20)
plt.grid()
plt.show()
plt.figure(figsize=[6,3])
plt.plot(T2,acc_rate2,label='L=2')
plt.plot(T4,acc_rate4,label='L=4')
plt.plot(T6,acc_rate6,label='L=6')
plt.xlabel('T',fontsize=20)
plt.ylabel(r'Acc_ratio',fontsize=20)
plt.grid()
plt.show()


