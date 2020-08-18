import numpy as np
import matplotlib.pyplot as plt

gold_standard = np.load('Langevin_Machine_Learning/init/N4_T0.85_ts0.001_vv_1000sampled.npy')
tau_large = np.load('Langevin_Machine_Learning/init/N4_T0.85_ts0.01_vv_1000sampled.npy')

print(gold_standard.shape)
print(tau_large.shape)



#gs_q_ = gold_standard[0,:30,:4]
gs_q_ = gold_standard[0]
# print(gs_q_[:21].shape)
# print('initial',gs_q_[0])
# print('10step',gs_q_[10])
# print('20step',gs_q_[20])
gs_q = gs_q_[0::10]  #[0,10,20,30 ...,]

# print('initial',gs_q[0])
# print('10step',gs_q[1])
# print('20step',gs_q[2])
#q    = tau_large[0,:3,:4]
q    = tau_large[0]

#gs_p_ = gold_standard[1,:30,:4]
gs_p_ = gold_standard[1]
gs_p = gs_p_[0::10]  #[0,10,20,30 ...,]
#p    = tau_large[1,:3,:4]
p    = tau_large[1]

print('----gs_q----')
print(gs_q.shape)
# print(gs_q)

# print(gs_q[:,0].shape) # sample 1
# print(gs_q[0,1]) # initial in sample 1
# print(q[:,0].shape)  # sample 1
# print(q[0,1]) # initial in sample 1

N_particle =2
sample = gs_q.shape[1]
print('sample',sample)
iterations = 1000
time_step = 0.01

# x
del_q =  np.power((gs_q - q),2)
del_p =  np.power((gs_p - p),2)
# print('----gs_q----')
# print(gs_q)
# print('----q----')
# print(q)
# print('--gs_q - q--')
# print(gs_q - q)
# print('----del_q----')
# print(del_q)

# print('----gs_p----')
# print(gs_p)
# print('----p----')
# print(p)
# print('--gs_p - p--')
# print(gs_p - p)
# print('----del_p----')
# print(del_p)

avg_del_q_sample = np.sum(del_q,axis=1) / sample
# print('----avg_del_q_sample----')
# print(avg_del_q_sample)

avg_del_q_sample_particle = np.sum(avg_del_q_sample,axis=1) / N_particle
print('----avg_del_q_sample_particle----')
print(avg_del_q_sample_particle)

avg_del_p_sample = np.sum(del_p,axis=1) / sample
# print('----avg_del_p_sample----')
# print(avg_del_p_sample)

avg_del_p_sample_particle = np.sum(avg_del_p_sample,axis=1) / N_particle
print('----avg_del_p_sample_particle----')
print(avg_del_p_sample_particle)

avg_del_qp_sample_particle = avg_del_q_sample_particle + avg_del_p_sample_particle
print('----avg_del_qp_sample_particle----')
print(avg_del_qp_sample_particle)

fig = plt.figure()
t = np.arange(0., iterations * time_step + time_step, time_step)

fig.add_subplot(3,2,1)
plt.title('q coordinate x iterations {}; time step {}'.format(iterations,time_step))
plt.plot(t,avg_del_q_sample_particle[:,0], label = 'Distance metric')

fig.add_subplot(3,2,2)
plt.title('q coordinate y iterations {}; time step {}'.format(iterations,time_step))
plt.plot(t,avg_del_q_sample_particle[:,1], label = 'Distance metric')

fig.add_subplot(3,2,3)
plt.title('p coordinate x iterations {}; time step {}'.format(iterations,time_step))
plt.plot(t,avg_del_p_sample_particle[:,0], label = 'Distance metric')

fig.add_subplot(3,2,4)
plt.title('p coordinate y iterations {}; time step {}'.format(iterations,time_step))
plt.plot(t,avg_del_p_sample_particle[:,1], label = 'Distance metric')

fig.add_subplot(3,2,5)
plt.title('qp coordinate x iterations {}; time step {}'.format(iterations,time_step))
plt.plot(t,avg_del_qp_sample_particle[:,0], label = 'Distance metric')

fig.add_subplot(3,2,6)
plt.title('qp coordinate y iterations {}; time step {}'.format(iterations,time_step))
plt.plot(t,avg_del_qp_sample_particle[:,1], label = 'Distance metric')

plt.show()
