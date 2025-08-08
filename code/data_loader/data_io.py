import torch

class data_io:

    # standardize shape of qpl_list is [nsamples, (q, p, boxsize), trajectory length, nparticle, DIM]
    # standardize shape of q_list  is [nsamples,         trajectory length, nparticle, DIM]
    # ================================================
    @staticmethod
    def read_trajectory_qpl(filename):
        ''' given a filename, read the qp paired pts trajectory

        returns
        load the dictionary and then return all
        shape of qp_list is [nsamples, (q, p,l), trajectory length, nparticle, DIM]
        '''

        data = torch.load(filename,weights_only=False) # 20250808

        qpl_list = data['qpl_trajectory']
        tau_short = float(data['tau_short'])
        tau_long  = float(data['tau_long'])

        return qpl_list, tau_short, tau_long

    # ================================================
    @staticmethod
    def write_trajectory_qpl(filename, qpl_trajectory, tau_short = -1, tau_long = -1):
        ''' write filename for qp_trajectory

        Parameters
        ----------
        filename : string
        nparticle : int
        boxsize  : float
        qpl_trajectory : torch.tensor
                  tensor of (q,p, boxsize) states
                  two kinds of shapes:
                  shape is [nsamples, (q, p, boxsize), trajectory length, nparticle, DIM]
                  OR
                  shape is [nsamples, (q, p, boxsize), 1, nparticle, DIM]
        tau_short tau_long : float or int
                  default is negative values for MC output
                  positive values for MD output
        returns
        save multiple components, organize them in a dictionary
        '''

        data = { 'qpl_trajectory':qpl_trajectory, 'tau_short':tau_short, 'tau_long': tau_long }

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

    # ================================================
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
    # ================================================
    @staticmethod
    def random_shuffle(qpl_list,tau_long,tau_short):

        nsamples = qpl_list.shape[0]
        rand_idx = torch.randperm(nsamples)
        qpl_shuffle = qpl_list[rand_idx]
        #tau_long_shuffle = tau_long[rand_idx] == tau long is one scalar number
        #tau_short_shuffle = tau_short[rand_idx]

        return qpl_shuffle
    # ================================================



if __name__=='__main__':

    filename = '../../data_sets/n16anyrholt0.1stps_100pts.pt'
    qpl_list,tau_long,tau_short = data_io.read_trajectory_qpl(filename)

    print('shape of qpl_list [nsamples, (q,p,l)-index, ', end='')
    print('nsteps label (initial,1, 4, 8, 16 steps), nparticles, dimension')

    print('qpl_list shape ',qpl_list.shape)

    print('qpl_list ',qpl_list[:2])

    qpl_sh = data_io.random_shuffle(qpl_list,tau_long,tau_short)

    print('qpl_sh ',qpl_sh[:2])



