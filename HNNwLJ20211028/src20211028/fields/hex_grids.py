import torch
import matplotlib.pyplot as plt

class hex_grids:

    ''' grids class to help make grids from each particle position'''

    _obj_count = 0

    def __init__(self, b, ngrid):
        '''
        Parameters
        ----------
        b : grid interval      b = rij = 0.4 * sigma => potential ~ 237442.02
        girds_18center   : hexagonal grids is 18 points at position (0,0) as center
        '''

        hex_grids._obj_count += 1
        assert(hex_grids._obj_count == 1), type(self).__name__ + ' has more than one object'

        if ngrid == 6 :
            gridb = torch.tensor([[-b *0.5, -b],[-b *0.5, b], [-b, 0.], [b, 0.], [b *0.5, -b], [b *0.5, b]] )
            self.grids_ncenter = gridb
            # grids_ncenter.shape is [6, 2]

        if ngrid == 18 :
            gridb = torch.tensor([[-b * 0.5, -b], [-b * 0.5, b], [-b, 0.], [b, 0.], [b * 0.5, -b], [b * 0.5, b]])
            girdhb = torch.tensor([[0., 2*b], [0., -2*b], [-b *1.5, b],  [-b *1.5,-b], [b *1.5,b], [b *1.5,-b]] )
            grid2b = 2*gridb
            self.grids_ncenter = torch.cat((grid2b,gridb,girdhb))
            # grids_ncenter.shape is [18, 2]

        print('grids initialized : gridL ',b, 'ngrid ', ngrid)

    def make_grids(self, phase_space):
        '''
        make_grids function to shift 6 or 18 grids points at (0,0) to each particle position as center

        :return
        shift grids to particle position as center
                shape is [nsamples, nparticle, grids, DIM=(x,y)]
        '''

        q_list = phase_space.get_q()
        boxsize = phase_space.get_l_list()
        # boxsize.shape is [nsamples, nparticle, DIM]

        boxsize = torch.unsqueeze(boxsize, dim=2)
        # boxsize.shape is [nsamples, nparticle, 1, DIM]
        boxsize = boxsize.repeat_interleave(self.grids_ncenter.shape[0],dim=2)

        q_list = torch.unsqueeze(q_list,dim=2)
        # q_list.shape is [nsamples, nparticle, 1, DIM=(x coord, y coord)]

        grids_shift = self.grids_ncenter + q_list
        # grids_ncenter.shape is [18, 2] + [nsamples, nparticle, 1, DIM] => [nsample, nparticle, 18, DIM=2]
        # grids_shift.shape is [nsamples, nparticle, 18, DIM=(x,y)]

        phase_space.adjust_real_q(grids_shift,boxsize) # pbc - for grids

        return  grids_shift


    def show_grids_nparticles(self, grids_list, q_list,boxsize):

        for i in range(1): # show two samples

            plt.title('sample {}'.format(i))
            plt.xlim(-boxsize/2, boxsize/2)
            plt.ylim(-boxsize / 2, boxsize / 2)
            plt.plot(grids_list[:,:,0], grids_list[:,:,1], marker='.', color='k', linestyle='none', markersize=12)
            plt.plot(q_list[ :, 0], q_list[ :, 1], marker='x', color='r', linestyle='none', markersize=12)
            plt.show()
            plt.close()