import torch
from utils.pbc import pbc
from utils.mydevice import mydevice

# check grid=1
import sys
# sys.path.append('../../')
# from utils.pbc import pbc
# from ML.networks.mockPWnet import mockPWnet
# from utils.system_logs import system_logs
# from utils.mydevice import mydevice
# from ML.LLUF.SingleGrid import SingleGrid
# import PrepareData

# for each particle, use pwnet to
# generate features for each point 
# on a hex grid centered at the particle
#
# we can have different layers of hex grids

class PhiFeatures:

    def __init__(self,grid_object,net): #b_list,a_list,net):
        self.grid_object = grid_object
        self.net = net
        self.mask = None

        self.dim = self.grid_object.dim # 20250807

        ## 20250803 phi feature output dim here. ngrid * pwnet output dim

    # ===================================================
    def __call__(self, q_list, l_list):  # make dqdp for n particles
        # q_list shape [nsamples, nparticles, DIM = 2 or 3] # 20250807 now we only implement 3d system.
        # l_list shape [nsamples, nparticles, DIM = 2 or 3]
        # make all grid points        self.nsamples, self.nparticles, self.DIM = q_list.shape
        uli_list = self.grid_object(q_list, l_list)  # position at grid points
        # uli_list.shape is [nsamples, nparticles * ngrids, DIM=(x,y) or (x,y,z)]
        q_var = self.gen_qvar(q_list, l_list, uli_list, self.net, self.grid_object.ngrids)  # force fields
        # q_var.shape  [nsamples, npartice, ngrids * DIM =2 or 3], ngrids=6 or 12, DIM=2 or 3

        return q_var

    # ===================================================
    def make_mask(self,nsamples,nparticles,dim):
        # mask to mask out only centered at i-th particle
        # and then use to make zero of pair-wise which interact with itself
        self.mask = torch.ones([nsamples,nparticles,nparticles,dim],device=mydevice.get())
        dia = torch.diagonal(self.mask,dim1=1,dim2=2) # [nsamples, dim =2 or 3, nparticles]
        dia.fill_(0.0)
        return self.mask
    # ===================================================
    def gen_qvar(self, q_list, l_list, uli_list, pwnet, ngrids): 
        # uli_list.shape is [nsamples, nparticles * ngrids, DIM=2 or 3] # 20250807 now we only implement 3d system.
        nsamples, nparticles, DIM = q_list.shape
        _qvar = self.qvar(q_list, l_list, uli_list, pwnet, ngrids)
        # shape is [ nsamples, nparticles * ngrids, DIM =2 or 3]
        _gen_qvar = _qvar.view(nsamples, nparticles, -1)
        # shape is [ nsamples, nparticles, ngrids * DIM ]

        return _gen_qvar
    # ===================================================
    def qvar(self, q_list, l_list, uli_list, pwnet, ngrids):  # uli_list = grid center position
        # uli_list.shape is [nsamples, nparticles * ngrids, DIM=2 or 3] # 20250807 now we only implement 3d system.

        nsamples, nparticles, dim = q_list.shape

        l_list = torch.unsqueeze(l_list, dim=2)
        # l_list.shape is [nsamples, nparticles, 1, DIM = 2 or 3]

        l_list = l_list.repeat_interleave(uli_list.shape[1], dim=2)
        # l_list.shape is [nsamples, nparticles, nparticles * ngrids, DIM= 2 or 3]

        _, dq_sq = self.dpair_pbc_sq(q_list, uli_list, l_list)
        # d_sq.shape is [nsamples, nparticles, nparticles * ngrids]
        #print(dq_sq)
        dq_sq = dq_sq.view(nsamples * nparticles * nparticles * ngrids, 1)  # dq^2
        # dq_sq.shape is [nsamples * nparticles * nparticles * ngrids, 1]

        pair_pwnet = pwnet(dq_sq)
        # pair_pwnet.shape = [batch, 2]

        pair_pwnet = pair_pwnet.view(nsamples, nparticles, nparticles * ngrids, -1)
        # shape is [nsamples, nparticles, nparticles * ngrids, DIM]
        #print(pair_pwnet[:,:,:,0])
        #print(pair_pwnet[:, :, :, 1])

        if self.grid_object.ngrids == 1:
            pair_pwnet = pair_pwnet * self.make_mask(nsamples,nparticles,dim)

        #print(pair_pwnet[:, :, :, 0])
        #print(pair_pwnet[:, :, :, 1])

        # pair_pwnet = self.zero_qvar(pair_pwnet, nsamples, nparticles, ngrids)
        ## pair_pwnet.shape is [nsamples, nparticles, nparticles*ngrids, DIM]

        q_var = torch.sum(pair_pwnet, dim=1)  # np.sum axis=2 j != k ( nsamples-1)
        # q_var.shape is [nsamples, nparticles * ngrids, DIM=2]
        #print('sum of them...')
        #print(q_var)

        return q_var

    # ===================================================
    def dpair_pbc_sq(self, q, uli_list, l_list):  #

        # all list dimensionless
        q_state = torch.unsqueeze(q, dim=2)
        # shape is [nsamples, nparticles, 1, DIM=(x,y)]

        uli_list = torch.unsqueeze(uli_list, dim=1)
        # shape is [nsamples, 1, nparticles * ngrids, DIM=(x,y)]

        paired_grid_q = uli_list - q_state
        # paired_grid_q.shape is [nsamples, nparticles, nparticles * ngrids, DIM]
        #print('diff',paired_grid_q)

        pbc(paired_grid_q, l_list)
        #print('pbc', paired_grid_q)

        dd = torch.sum(paired_grid_q * paired_grid_q, dim=-1)
        # dd.shape is [nsamples, nparticles, nparticles * ngrids]
        #print('sq sum',dd)

        return paired_grid_q, dd

    # ===================================================
    #def zero_qvar(self, pair_pwnet, nsamples, nparticles, ngrids):
    #    # pair_pwnet shape is, [nsamples * nparticles * nparticles * ngrids, 2]
    #
    #    _, DIM = pair_pwnet.shape
    #    pair_pwnet1 = pair_pwnet.view(nsamples, nparticles, nparticles, ngrids, DIM)
    #    # make_zero_phi.shape is [nsamples, nparticles, nparticles, ngrids, DIM]
    #
    #    pair_pwnet2 = pair_pwnet1 * self.mask
    #
    #    pair_pwnet3 = pair_pwnet2.view(nsamples, nparticles, nparticles * ngrids, DIM)
    #    # shape is [nsamples, nparticles, nparticles*ngrids, DIM]
    #
    #    return pair_pwnet3
    #
    #

if __name__=='__main__':

    _ = mydevice()
    _ = system_logs(mydevice)
    system_logs.print_start_logs()

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(34952)

    qt1 = (torch.randn(1,4,2) - 0.5) * 3
    qt2 = qt1 * 0.97
    qt3 = qt2 * 0.97

    pt1 = (torch.randn(1,4,2) - 0.5) * 3
    pt2 = pt1 * 0.97
    pt3 = pt2 * 0.97

    print('qt1..... ')
    print(qt1)
    #print('pt1..... ')
    #print(pt1)
    print('qt2..... ')
    print(qt2)
    #print('pt2..... ')
    #print(pt2)
    print('qt3..... ')
    print(qt3)
    #print('pt3.....')
    #print(pt3)

    qt = torch.stack((qt1,qt2,qt3),dim=1)
    pt = torch.stack((pt1, pt2, pt3), dim=1)
    l_list = torch.ones(qt1.shape) * 3
    #t= torch.stack((qt,pt,l_list), dim=1)
    #print(t.shape)
    # t.shape = [nsamples, (q,p,boxsize)=3, trajectory, nparticles, DIM]

    q_traj = qt.permute(1, 0, 2, 3)
    p_traj = pt.permute(1, 0, 2, 3)

    q_traj = mydevice.load(q_traj)
    p_traj = mydevice.load(p_traj)
    l_list = mydevice.load(l_list)

    prepare_data_net = mydevice.load(mockPWnet(1, 2, 128, 'tanh'))
    grid_object = SingleGrid()

    prepare_data_obj = PrepareData.PrepareData(prepare_data_net, grid_object)

    nsamples, nparticles, dim = qt1.shape
    #print(prepare_data_obj)

    #PhiFeatures.make_mask(nsamples, nparticles, dim)

    q_traj_list = list(q_traj)
    p_traj_list = list(p_traj)

    q_cur = q_traj[-1]
    p_cur = p_traj[-1]

    q_input_list = []
    p_input_list = []

    for q, p in zip(q_traj_list, p_traj_list):
        q_input_list.append(prepare_data_obj.prepare_q_feature_input(q,l_list))
        p_input_list.append(prepare_data_obj.prepare_p_feature_input(q, p, l_list))

