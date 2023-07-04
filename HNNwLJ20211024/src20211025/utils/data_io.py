import torch

class data_io:

    # standardize shape of qp_list is [nsamples, (q, p), trajectory length, nparticle, DIM]
    # standardize shape of q_list  is [nsamples,         trajectory length, nparticle, DIM]
    # ================================================
    @staticmethod
    def read_trajectory_qp(filename):
        ''' given a filename, read the qp paired pts trajectory

        returns
        load the dictionary and then return all
        shape of qp_list is [nsamples, (q, p), trajectory length, nparticle, DIM]
        '''

        data = torch.load(filename)

        qp_list = data['qp_trajectory']
        tau_short = data['tau_short']
        tau_long  = data['tau_long']
        boxsize = data['boxsize']

        return qp_list, tau_short, tau_long, boxsize

    # ================================================
    @staticmethod
    def write_trajectory_qp(filename, qp_trajectory, boxsize, tau_short = -1, tau_long = -1):
        ''' write filename for qp_trajectory

        Parameters
        ----------
        filename : string
        nparticle : int
        boxsize  : float
        qp_trajectory : torch.tensor
                  tensor of (q,p) states
                  two kinds of shapes:
                  shape is [nsamples, (q, p), trajectory length, nparticle, DIM]
                  OR
                  shape is [nsamples, (q, p), 1, nparticle, DIM]
        tau_short tau_long : float or int
                  default is negative values for MC output
                  positive values for MD output
        returns
        save multiple components, organize them in a dictionary
        '''

        data = { 'qp_trajectory':qp_trajectory, 'tau_short':tau_short, 'tau_long': tau_long, 'boxsize' : boxsize }

        torch.save(data, filename)

    # ================================================
    @staticmethod
    def read_crash_info(filename): 
        ''' given a filename, read the iteration that get crash data
        and the number of crash data at the iteration

        returns
        load the dictionary and then return all
        crash_niter.shape, scrash_nct is [n]
        '''
        data = torch.load(filename)
        crash_niter = data['crash_niter']
        crash_nct = data['crash_nct']

        return crash_niter, crash_nct

    # ================================================
    @staticmethod
    def write_crash_info(filename, crash_niter, crash_nct): 
        ''' write filename for crash info

        Parameters
        ----------
        filename : string
        crash_niter : int
        crash_nct : int

        returns
        save iteration that get crash data and the number of crash data  at the iteration
        '''
        crash_info = {'crash_niter': crash_niter, 'crash_nct': crash_nct}
        torch.save(crash_info, filename)

    @staticmethod
    def write_param_dist(filename, weight, bias):
        ''' write filename for parameters weights and bias

        Parameters
        ----------
        filename : string
        weight : float
        bias : float

        returns
        save weights and bias to plot distribution of parameters that save during training
        '''
        w_b_dis = {'weights': weight, 'bias': bias}
        torch.save(w_b_dis, filename)
