from hamiltonian.hamiltonian            import hamiltonian
from hamiltonian.lennard_jones          import lennard_jones
from hamiltonian.kinetic_energy         import kinetic_energy
from utils.data_io                      import data_io  
from utils.get_paired_distance_indices  import get_paired_distance_indices
import torch
import itertools
import numpy as np

class HNN_base(hamiltonian):

    _obj_count = 0

    def __init__(self, net1, net2):

        super().__init__()

        #HNN_base._obj_count += 1
        #assert (HNN_base._obj_count == 1), type(self).__name__ + " has more than one object"

        self.net1 = net1   # network for dHdq1
        self.net2 = net2   # network for dHdq2

        # append term to calculate dHdq
        super().append(lennard_jones())
        super().append(kinetic_energy())

        # dhdq1
        self.sumRx1 = 0. # set 0 to sum Rx over batch size
        self.sumRy1 = 0. # set 0 to sum Ry over batch size
        self.R_cnt1 = 0. # count the number of batch size

        # dhdq2
        self.sumRx2 = 0. # set 0 to sum Rx over batch size
        self.sumRy2 = 0. # set 0 to sum Ry over batch size
        self.R_cnt2 = 0. # count the number of batch size

        print('HNN_base initialized')

        self.dt=0 # set 0

    # ===================================================
    def calculate_RxRy_ratio(self,noMLdHdq,MLdHdq): 
        ''' calculate ratio btw nomldhdq and mldhdq
        pass nomldhd and mldhdq and detach and convert tensor into numpy array
        these tensors don't need to calc gradients so that detach from graph

        Parameters
        ----------
        noMLdHdq.shape, MLdHdq.shape = [nsamples, nparticle, DIM]

        :return
        Rx.shape , Ry.shape = [nsamples, nparticle]
        '''
        Rx = abs(MLdHdq[:,:,0]) / ( abs(noMLdHdq[:,:,0]) + abs(MLdHdq[:,:,0]) )
        Ry = abs(MLdHdq[:,:,1]) / ( abs(noMLdHdq[:,:,1]) + abs(MLdHdq[:,:,1]) )
        # Rx.shape, Ry.shape = [nsamples, nparticle]
        Rx = Rx.detach().numpy()
        Ry = Ry.detach().numpy()
        return Rx, Ry
    # ===================================================
    def sum_RxRy_dhdq1(self, Rx, Ry):
        ''' From net1, accumulate mean Rx, Ry at every batch size till get one epoch
        and count the number of batch size '''

        self.sumRx1 += np.mean(Rx)   # accumulate mean over nsamples over nparticle
        self.sumRy1 += np.mean(Ry)   # accumulate mean over nsamples over nparticle
        self.R_cnt1 += 1. # count the number of batch size;
    # ===================================================
    def sum_RxRy_dhdq2(self, Rx, Ry):
        ''' From net2, accumulate mean Rx, Ry at every batch size till get one epoch
        and count the number of batch size '''

        self.sumRx2 += np.mean(Rx)   # accumulate mean over nsamples over nparticle
        self.sumRy2 += np.mean(Ry)   # accumulate mean over nsamples over nparticle
        self.R_cnt2 += 1. # count the number of batch size;
    # ===================================================
    def get_RxRy_dhdq1(self):
        ''' get avg Rx, Ry at one epoch using net 1,
        reset sum Rx, Ry and counting the number of batch size '''

        aveRx = self.sumRx1 / self.R_cnt1 # calculate average over nsamples
        aveRy = self.sumRy1 / self.R_cnt1  # calculate average over nsamples
        self.sumRx1 = 0. # reset
        self.sumRy1 = 0. # reset
        self.R_cnt1 = 0. # reset
        return aveRx, aveRy
    # ===================================================
    def get_RxRy_dhdq2(self):
        ''' get avg Rx, Ry at one epoch using net 2,
        reset sum Rx, Ry and counting the number of batch size '''

        aveRx = self.sumRx2 / self.R_cnt2 # calculate average over nsamples
        aveRy = self.sumRy2 / self.R_cnt2 # calculate average over nsamples
        self.sumRx2 = 0. # reset
        self.sumRy2 = 0. # reset
        self.R_cnt2 = 0. # reset
        return aveRx, aveRy
    # ===================================================
    def show_total_nn_time(self):
        nn_dt = self.dt
        self.dt = 0
        return nn_dt
    # ===================================================
    def set_tau_long(self, tau_long):
        self.tau_long = tau_long
    # ===================================================
    def requires_grad_false(self):
        ''' use when predict trajectory as make requires_grad=False
         it means not update weights and bias'''

        for param in self.net1.parameters():
            param.requires_grad = False

        for param in self.net2.parameters():
            param.requires_grad = False
    # ===================================================
    def check_grad_minmax(self):  
        ''' load saved parameter from trained model and original data as no crash data
        use to check param gradient min-max and save param grad after backward '''
        gradw = []
        gradb = []
        for n in self.get_netlist():
            for name, param in n.named_parameters():
                if 'weight' in name:
                    gradw.append(param.grad.view(-1))
                if 'bias' in name:
                    gradb.append(param.grad.view(-1))
        gradw = torch.cat(gradw)
        gradb = torch.cat(gradb)

        data_io.write_param_dist('../data/gen_by_ML/nocrash_retrain/gradw_gradb.pt', gradw, gradb)

        print('grad w max', max(gradw),'min', min(gradw))
        print('grad b max', max(gradb), 'min', min(gradb))
    # ===================================================
    def check_param_minmax(self): 
        ''' load saved parameter from trained model and original data as no crash data
        use to check parameters min-max and save param after backward '''
        ws = []
        bs = []
        for n in self.get_netlist():
            for name, param in n.named_parameters():
                if 'weight' in name:
                    ws.append(param.view(-1))
                if 'bias' in name:
                    bs.append(param.view(-1))
        ws = torch.cat(ws)
        bs = torch.cat(bs)

        data_io.write_param_dist('../data/gen_by_ML/nocrash_retrain/w_b.pt', ws, bs)

        print('w max', max(ws),'min', min(ws))
        print('b max', max(bs), 'min', min(bs))
    # ===================================================
    def get_netlist(self): # nochange
        return [self.net1,self.net2]
    # ===================================================
    def net_parameters(self): # nochange
        ''' To give multiple parameters to the optimizer,
            concatenate lists of parameters from two models '''
        return itertools.chain(self.net1.parameters(), self.net2.parameters())
    # ===================================================
    def train(self):
        '''pytorch network for training'''
        self.net1.train()
        self.net2.train()
    # ===================================================
    def eval(self):
        ''' pytorch network for eval '''
        self.net1.eval()
        self.net2.eval()
    # ===================================================
    def potential_rep(self, q_list, phase_space):
        ''' function to use for tune the function by adding an additional penalty term in loss'''

        boxsize = phase_space.get_l_list()
        nsamples, nparticle, DIM = q_list.shape
        # shape = [nsamples, nparticle, DIM]

        q_list = q_list / boxsize #  dimensionless
        _, dr = phase_space.paired_distance_reduced(q_list, nparticle, DIM)

        s12 = 1 / pow(dr, 12)
        # s12.shape = [nsamples, nparticle, nparticle-1]

        boxsize = torch.mean(boxsize, dim=-1)
        # boxsize.shape is [nsamples, nparticle]
        boxsize = torch.unsqueeze(boxsize,dim=-1)
        # boxsize.shape is [nsamples, nparticle,1]

        term_dim = torch.sum(4 / pow(boxsize, 12) * s12, dim=-1)
        # term_dim.shape is [nsamples, nparticle]

        u_rep =  torch.sum(term_dim, dim=-1) * 0.5
        # term.shape is [nsamples]

        u_rep_max = torch.max(u_rep)

        return u_rep_max


    # ===================================================
    def print_grad(self):
        self.net1.print_grad()
        self.net2.print_grad()
