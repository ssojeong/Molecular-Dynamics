import torch
from einops import rearrange, repeat

from utils.pbc import pbc
from utils.mydevice import mydevice
from ML.LLUF.AtomPairIndexer import AtomPairIndexer


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

    def __init__(self, grid_object, net):   # b_list,a_list,net):
        self.grid_object = grid_object
        self.net = net
        self.mask = None

        self.dim = self.grid_object.dim     # 20250807
        self.atom_pair_indexer = AtomPairIndexer(n_mol=8)

        # 20250803 phi feature output dim here. ngrid * pwnet output dim

    # ===================================================
    def __call__(self, q_list, l_list):  # make dqdp for n particles
        # make all grid points        self.nsamples, self.nparticles, self.DIM = q_list.shape
        uli_list = self.grid_object(q_list, l_list)  # position at grid points
        # uli_list.shape is [nsamples, nparticles * ngrids, DIM=(x,y) or (x,y,z)]
        q_var = self.gen_qvar(q_list, l_list, uli_list, self.net, self.grid_object.ngrids)  # force fields
        # q_var.shape  [nsamples, npartice, ngrids * DIM =2 or 3], ngrids=6 or 12, DIM=2 or 3

        return q_var

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

    def qvar(self, q_list, l_list, uli_list, pwnet, ngrids):        # revised by LW
        """
        Compute q_var using einops.

        Shapes (3D system):
          q_list   : (ns, np, 3)                      # particle positions
          l_list   : (ns, np, 3)                      # particle lengths (per particle vector, e.g., box or scale)
          uli_list : (ns, np*ngrids, 3)               # grid center positions per "j" particle
          ngrids   : int
          pwnet    : module mapping (B', C, G) -> (B', 3, G), where C=8 channels and G=ngrids

        Internals:
          dq_sq            : (ns, np, np*ngrids)      # squared distances per (i, j*grid)
          after one-hot    : (ns, np, np, 8, ngrids)
          flattened to     : ((ns*np*np), 8, ngrids)
          pwnet output     : ((ns*np*np), 3, ngrids)
          reshaped to      : (ns, np, np, ngrids, 3)
          merged (j,grid)  : (ns, np, np*ngrids, 3)
          sum over i       : (ns, np*ngrids, 3)       # q_var
        """
        ns, np, dim = q_list.shape
        assert dim == 3, "Currently only 3D is implemented."

        # Expand l_list along the (j, grid) axis to match uli_list
        # l_list: (ns, np, 3) -> (ns, np, np*ngrids, 3)
        l_list_expanded = repeat(l_list, 'ns i d -> ns i (j g) d', j=np, g=ngrids, d=3)

        # Compute distances (PBC-aware) → returns (_, dq_sq)
        # Expect: dq_sq: (ns, np, np*ngrids)
        q_list = pbc(q_list, l_list)
        _, dq_sq = self.dpair_pbc_sq(q_list, uli_list, l_list_expanded)

        # Reshape dq_sq to (ns, i, j, g)
        dq_sq_4d = rearrange(dq_sq, 'ns i (j g) -> ns i j g', j=np, g=ngrids)

        # One-hot by atom-pair channel: (ns, i, j, C=8, g)
        dq_sq_one_hot = self.atom_pair_indexer.fill_tensor(dq_sq_4d)  # expects (B,N,N,G) -> (B,N,N,8,G)

        # Flatten (ns, i, j) -> (ns*i*j) for pwnet; (ns*i*j, 8, g)
        x = rearrange(dq_sq_one_hot, 'ns i j c g -> (ns i j) c g', c=self.atom_pair_indexer.num_channels)
        # Apply pairwise network: (ns*i*j, 3, g)
        y = pwnet(x)
        assert y.shape == (ns * np * np, 3, ngrids), f"pwnet output shape mismatch: {y.shape}"

        # Back to (ns, i, j, g, 3)
        y_5d = rearrange(y, '(ns i j) c g -> ns i j g c', ns=ns, i=np, j=np)

        # Merge (j, g) → (j*g) and sum over i (axis=1)
        pair_pwnet = rearrange(y_5d, 'ns i j g c -> ns i (j g) c')
        q_var = pair_pwnet.sum(dim=1)  # (ns, j*g, 3)

        return q_var  # shape: (ns, np*ngrids, 3)

    # ===================================================
    def dpair_pbc_sq(self, q, uli_list, l_list):  #

        # all list dimensionless
        q_state = torch.unsqueeze(q, dim=2)
        atom_pair_dis = torch.norm(q[:, None] - q[:, :, None], dim=-1)
        print(q.shape, atom_pair_dis.shape)
        for i in range(atom_pair_dis.size(1)):
            atom_pair_dis[:, i, i] = 1
        print('atom pair distance min:', atom_pair_dis.min(), 'max:', atom_pair_dis.max())
        # shape is [nsamples, nparticles, 1, DIM=(x,y)]

        uli_list = torch.unsqueeze(uli_list, dim=1)
        # shape is [nsamples, 1, nparticles * ngrids, DIM=(x,y)]

        print(q_state.shape, uli_list.shape)
        paired_grid_q = uli_list - q_state
        # paired_grid_q.shape is [nsamples, nparticles, nparticles * ngrids, DIM]
        # print('diff',paired_grid_q)

        print('before pbc', paired_grid_q.min().item(), paired_grid_q.abs().max().item())
        pbc(paired_grid_q, l_list)
        print('after pbc', paired_grid_q.min().item(), paired_grid_q.abs().max().item())

        r_square = torch.sum(paired_grid_q * paired_grid_q, dim=-1)

        r_norm = torch.norm(paired_grid_q, dim=-1)
        min_val, min_idx = torch.min(r_norm.reshape(-1), dim=0)  # along last axis (288)
        min_row = min_idx // 288
        min_col = min_idx % 288
        print("min val", min_val, "min idx", min_row, min_col)
        print(r_norm[0, min_row, min_col])
        print('q_state', q_state[0, min_row, 0, :])
        print('u_list', uli_list[0, 0, min_col, :])
        print('u_center', q_state[0, min_col//12, 0, :])
        a = q_state[0, min_row, 0, :] - uli_list[0, 0, min_col, :]
        b = q_state[0, min_row, 0, :] - q_state[0, min_col//12, 0, :]
        c = uli_list[0, 0, min_col, :] -q_state[0, min_col//12, 0, :]
        print(torch.norm(a).item(), torch.norm(b).item(), torch.norm(c).item())

        print('atom grid dist', r_norm.min().item(), r_norm.max().item())
        print('r_square', r_square.min().item(), r_square.max().item())
        assert torch.min(r_square) > 0.0004 * 0.9, f'Min value of dq_sq {torch.min(r_square).item()},'
        # dd.shape is [nsamples, nparticles, nparticles * ngrids]
        # print('sq sum',dd)

        return paired_grid_q, r_square


if __name__ == '__main__':

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
