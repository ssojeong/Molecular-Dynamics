import torch
from data_loader.data_io import data_io
from torch.utils.data import Dataset
import random
import numpy as np

# qp_list.shape = [nsamples, (q,p,l)=3, trajectory=2 (input,label), nparticle, DIM]
# ===========================================================
class torch_dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, filename, label_idx):
        """
        Args:
            filename (string): Numpy file for data and label
        """
        qpl_list, tau_short, tau_long = data_io.read_trajectory_qpl(filename)
        # qpl_list.shape = [nsamples, (q,p,boxsize)=3, trajectory (input,label), nparticle, DIM]

        self.qp_list_input   = qpl_list[:,:,0,:,:]
        self.qp_list_label   = qpl_list[:,:,label_idx,:,:] 
        self.data_boxsize   =  qpl_list[:,2,0,:,:]
        #data_boxsize.shape is [nsamples,nparticle,DIM]
        self.data_tau_short = tau_short
        self.data_tau_long  = float(tau_long)

    def __len__(self):
        ''' Denotes the total number of samples '''
        return self.qp_list_input.shape[0]

    def __getitem__(self, idx):
        ''' Generates one sample of data '''
        # Select sample
        if idx >= self.__len__():
            raise ValueError('idx ' + str(idx) +' exceed length of data: ' + str(self.__len__()))

        return self.qp_list_input[idx], self.qp_list_label[idx]

# ===========================================================
class my_data:
    def __init__(self,train_filename, val_filename, test_filename, n_chain=1, train_pts=0, val_pts=0, test_pts=0):

        n_chain_list = [1,4,8,16,32,64]
        if n_chain not in n_chain_list:
            print('invalid n_chain ')
            quit()

        if   n_chain == 1:  chain_index = 1 # index 0 is initial state
        elif n_chain == 4:  chain_index = 2 
        elif n_chain == 8:  chain_index = 3 
        elif n_chain == 16: chain_index = 4 
        elif n_chain == 32: chain_index = 5 
        else:               chain_index = 6 

        self.train_set = torch_dataset(train_filename, chain_index)
        self.val_set   = torch_dataset(val_filename,   chain_index)
        self.test_set  = torch_dataset(test_filename,  chain_index)

        # perform subsampling of data when specified
        # this is important when we need to perform quick debugging with
        # few data points
        if train_pts > 0:
            if train_pts > len(self.train_set):
                print('available ', len(self.train_set))
                raise ValueError("ERROR: request more than subspace set")
            self.train_set = self.sample(self.train_set, train_pts)

        if val_pts > 0:
            if val_pts > len(self.val_set):
                print('available ', len(self.val_set))
                raise ValueError("ERROR: request more than subspace set")
            self.val_set = self.sample(self.val_set, val_pts)

        if test_pts > 0:
            if test_pts > len(self.test_set):
                print('available ', len(self.test_set))
                raise ValueError("ERROR: request more than subspace set")
            self.test_set = self.sample(self.test_set, test_pts)

        print('my_data initialized : train_filename ',train_filename,' val_filename ',
               val_filename,' test_filename ',test_filename,' train_pts ',train_pts,
              ' val_pts ',val_pts,' test_pts ',test_pts)

    # ===========================================================
    # sample data_set with num_pts of points
    # ===========================================================
    def sample(self, data_set, num_pts):

        # perform subsampling of data when specified
        # this is important when we need to perform quick debugging with
        # few data points
        if num_pts > 0:
            if num_pts > len(data_set):
                print("error: request more than CIFAR10 set")
                print('available ',len(self.train_set))
                quit()
 
        data_set.qp_list_input = data_set.qp_list_input[:num_pts]
        data_set.qp_list_label = data_set.qp_list_label[:num_pts]

        return data_set

# ===========================================================
#
class data_loader:
    #  data loader upon this custom dataset
    def __init__(self,data_set, batch_size):

        self.data_set = data_set
        self.batch_size = batch_size

        use_cuda = torch.cuda.is_available()
        kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
        # num_workers: the number of processes that generate batches in parallel.
        print('kwargs ',kwargs, 'batch_size ', batch_size)

        self.train_loader = torch.utils.data.DataLoader(self.data_set.train_set,
                            batch_size=batch_size, shuffle=True, **kwargs)

        self.val_loader   = torch.utils.data.DataLoader(self.data_set.val_set,
                            batch_size=batch_size, shuffle=True, **kwargs)

        self.test_loader  = torch.utils.data.DataLoader(self.data_set.test_set,
                            batch_size=batch_size, shuffle=True, **kwargs)


