#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
from ..utils.data_util import data_loader
from ..Integrator.linear_integrator import linear_integrator
import os 
import numpy as np

class Hamiltonian_Dataset(Dataset):
    '''Custom class dataset for hamiltonian dataset'''
    
    def __init__(self, temperature : list, samples : int, mode : str, **kwargs):

        '''
        Parameters
        ----------
        temperature : list
            list of temperature to be used for training
        samples : int
            number of samples per temperature sampled
        mode : str
            only train/validation splitting
        **kwargs : configuration
        '''

        uppath = lambda _path, n : os.sep.join(_path.split(os.sep)[:-n])
        base_dir = uppath(__file__, 2)
        init_path = base_dir + '/init_config/'
        N_particle = kwargs['particle']
        DIM = kwargs['DIM']
        seed = kwargs.get('seed', 9372211)  # first: 937162211 second: 937111

        q_list, p_list = data_loader.loadp_q(init_path,
                                        temperature,
                                        samples,
                                        N_particle) # wrapper for dataloader

        _ratio = 1 # train-validation + valid splitting
        train_data, validation_data = data_loader.split_data(q_list, p_list, _ratio,  seed)

        #change the configuration as needed before integrate
        if mode == 'train' : # for training data
            curr_data = train_data
            print('generating the training data \n')
            del validation_data
        elif mode == 'validation':
            print('generating the validation data \n')
            curr_data = validation_data
            del train_data
        else :
            raise Exception('Mode not recognized, only train/validation')

        N = curr_data.shape[0] ;  kwargs['N'] = N # nsamples
        init_q = curr_data[:,0] ; kwargs['pos'] = init_q
        init_vel = curr_data[:,1] ; kwargs['vel'] = init_vel

        print("=init===")
        print(init_q)
        print(init_q.shape)
        print(kwargs)

        q_after, p_after = linear_integrator(**kwargs).integrate(multicpu=False) # using the integrator class

        init_p = init_vel * kwargs['m']
        q_after, p_after = q_after[-1], p_after[-1] # only take the last from the list

        # generate data by pairing up all particles - get delta_q
        delta_init_q = np.zeros( (N, N_particle , (N_particle - 1), DIM) )
        delta_init_p = np.zeros( (N, N_particle , (N_particle - 1), DIM) )

        for z in range(N):

            delta_init_q_, _ = kwargs['pb_q'].paired_distance_reduced(init_q[z]/kwargs['BoxSize']) #reduce distance
            delta_init_q_ = delta_init_q_ * kwargs['BoxSize']
            delta_init_p_, _ = kwargs['pb_q'].paired_distance_reduced(init_p[z]/kwargs['BoxSize']) #reduce distance
            delta_init_p_ = delta_init_p_ * kwargs['BoxSize']

            print('delta_init_q',delta_init_q_)
            print('delta_init_q',delta_init_q_.shape)
            print('delta_init_p',delta_init_p_)
            print('delta_init_p', delta_init_p_.shape)

            # delta_q_x, delta_q_y, t
            for i in range(N_particle):
                x = 0  # all index case i=j and i != j
                for j in range(N_particle):
                    if i != j:
                        print(i,j)
                        print(delta_init_q_[i,j,:])
                        delta_init_q[z][i][x] = delta_init_q_[i,j,:]
                        delta_init_p[z][i][x] = delta_init_p_[i,j,:]

                        x=x+1

        print('delta_init')
        print(delta_init_q)
        print(delta_init_p)

        # tau : #this is big time step to be trained
        # to add paired data array
        tau = np.array([kwargs['tau'] * kwargs['iterations']] * N_particle * (N_particle - 1))
        tau = tau.reshape(-1,N_particle,(N_particle - 1),1) # broadcasting

        print('concat')
        paired_data_ = np.concatenate((delta_init_q,delta_init_p),axis=-1) # N (nsamples) x N_particle x (N_particle-1) x (del_qx, del_qy, del_px, del_py)
        print(paired_data_)
        print(paired_data_.shape)
        paired_data = np.concatenate((paired_data_,tau),axis=-1) # nsamples x N_particle x (N_particle-1) x  (del_qx, del_qy, del_px, del_py, tau )
        paired_data = paired_data.reshape(-1,paired_data.shape[2],paired_data.shape[3]) # (nsamples x N_particle) x (N_particle-1) x  (del_qx, del_qy, del_px, del_py, tau )

        print(paired_data)
        print(paired_data.shape)

        init_q = np.expand_dims(init_q, axis=1)
        print('init_q',init_q.shape)
        init_q = init_q.reshape(-1,init_q.shape[2],init_q.shape[3])
        print('init_q',init_q.shape)

        init_p = np.expand_dims(init_p, axis=1)
        init_p = init_p.reshape(-1, init_p.shape[2], init_p.shape[3])

        q_after = np.expand_dims(q_after, axis=1)
        q_after = q_after.reshape(-1,q_after.shape[2],q_after.shape[3])
        p_after = np.expand_dims(p_after, axis=1)
        p_after = p_after.reshape(-1,p_after.shape[2],p_after.shape[3])

        init_data = np.concatenate((init_q, init_p),axis=1) # nsamples x (q,p) x N_particle  x DIM
        label = np.concatenate((q_after, p_after),axis=1) # nsamples x (q,p) x N_particle   x DIM

        print(label)
        print(label.shape)

        print('pair data',paired_data)
        print(paired_data.shape)
        print('init data',init_data)
        print(init_data.shape)
        print('label',label)
        print(label.shape)

        data = (init_data, paired_data)
        print('data',data)

        #populate the dataset 
        self._dataset = [] # change the data and label here
        self._dataset.append([data,label])

        print('dataset loaded')
        
    def __len__(self):
        return len(self._dataset)  # samples x data/label x q/p x num. of particles x DIM
    
    def __getitem__(self, idx):
        return (self._dataset[idx][0], self._dataset[idx][1])
