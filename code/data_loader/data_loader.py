import torch
from data_loader.data_io import data_io
from torch.utils.data import Dataset
from data_loader.check_load_data import check_load_data

# qp_list.shape = [nsamples, (q,p,l)=3, trajectory=2 (input,label), nparticle, DIM]
# ===========================================================
class torch_dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, filename, traj_len_idx, label_idx):
        """
        Args:
            filename (string): Numpy file for data and label
        """
        qpl_list,tau_short,_ = data_io.read_trajectory_qpl(filename) # returns qpl,tau_short,tau_long
        # qpl_list.shape = [nsamples, (q,p,boxsize)=3, trajectory, nparticles, DIM = 2 or 3]
        self.qpl_list_input   = qpl_list[:,:,0:traj_len_idx,:,:]
        # qp_list_input.shape = [nsamples, (q,p,boxsize)=3, trajectory, nparticles, DIM = 2 or 3]
        self.qpl_list_label   = qpl_list[:,:,traj_len_idx:label_idx+1,:,:]
        # qp_list_label.shape is [nsamples,nparticles,DIM = 2 or 3]
        self.data_boxsize   =  qpl_list[:,2,0,:,:]
        #data_boxsize.shape is [nsamples,nparticles,DIM = 2 or 3]
        #self.data_tau_short = tau_short
        #self.data_tau_long  = float(tau_long)

        self.check_load = check_load_data(self.qpl_list_input,self.qpl_list_label)
        #self.check_load.check(tau_short)

    def __len__(self):
        ''' Denotes the total number of samples '''
        return self.qpl_list_input.shape[0]

    def __getitem__(self, idx):
        ''' Generates one sample of data '''
        # Select sample
        if idx >= self.__len__():
            raise ValueError('idx ' + str(idx) +' exceed length of data: ' + str(self.__len__()))
        return self.qpl_list_input[idx], self.qpl_list_label[idx] 

# ===========================================================
class my_data:
    def __init__(self,train_filename,val_filename,test_filename,tau_long, window_sliding,
                      tau_traj_len,train_pts=0,val_pts=0,test_pts=0):

        traj_len_index = round(tau_traj_len/tau_long)
        label_index = int((traj_len_index - 1) + window_sliding)
        print('load my data ... traj len index', traj_len_index, 'window sliding', window_sliding, 'label index', label_index)

        print("load train set .........")
        self.train_set = torch_dataset(train_filename, traj_len_index, label_index)
        print("load valid set .........")
        self.val_set   = torch_dataset(val_filename, traj_len_index, label_index)
        print("load test set .........")
        self.test_set  = torch_dataset(test_filename, traj_len_index, label_index)

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
    def check_md_trajectory(self,q_init,p_init,q_final,p_final,l_list,neval,tau,nitr,append_strike):
        assert(self.train_set.check_load.md_trajectory(q_init,p_init,q_final,p_final,
                            l_list,neval,tau,nitr,append_strike )),'data_loader.py:82 error'
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

        data_set.qpl_list_input = data_set.qpl_list_input[:num_pts]
        data_set.qpl_list_label = data_set.qpl_list_label[:num_pts]

        return data_set

# ===========================================================
class data_loader:
    #  data loader upon this custom dataset
    def __init__(self,data_set, batch_size):

        self.data_set = data_set
        self.batch_size = batch_size

        use_cuda = torch.cuda.is_available()
        kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
        # num_workers: the number of processes that generate batches in parallel.
        print('kwargs ',kwargs, 'batch_size ', batch_size)

        self.train_loader = torch.utils.data.DataLoader(self.data_set.train_set,
                            batch_size=batch_size, shuffle=True, **kwargs)

        self.val_loader   = torch.utils.data.DataLoader(self.data_set.val_set,
                            batch_size=batch_size, shuffle=True, **kwargs)

        self.test_loader  = torch.utils.data.DataLoader(self.data_set.test_set,
                            batch_size=batch_size, shuffle=False, **kwargs)


