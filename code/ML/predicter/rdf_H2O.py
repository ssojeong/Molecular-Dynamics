import sys
sys.path.append('../../')

import torch
import math
import numpy as np
import matplotlib.pyplot as plt


# def compute_pairwise_distances_single_frame(positions):
#     """
#     Compute all pairwise distances between atoms for a single frame, excluding self-pairs
#     positions: tensor of shape (n_atoms, 3)
#     returns: flattened tensor of all pairwise distances
#     """
#     batch, n_atoms = positions.shape[0], positions.shape[1]
#
#     # Compute pairwise distance matrix using broadcasting
#     diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # shape: (batch, n_atoms, n_atoms, 3)
#     distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))  # shape: (batch, n_atoms, n_atoms)
#
#     # Extract upper triangle (excluding diagonal) to avoid duplicates and self-pairs
#     mask = torch.triu(torch.ones(n_atoms, n_atoms, dtype=torch.bool, device=positions.device), diagonal=1)
#     mask = mask.unsqueeze(0).expand(batch, -1, -1)
#     # print(mask, mask.shape)
#     pairwise_distances = distances[mask]
#     # print(distances.shape, pairwise_distances.shape)
#     return pairwise_distances


def minimum_image(vec, box):
    """Apply minimum image convention to displacement vectors."""
    return vec - box * torch.round(vec / box)


def pair_distribution_function(file_list, op_name):
    """Generates a pair-distribution function on given data"""

    n_bins = int((math.ceil(r_max) - math.floor(r_min)) / bin_width)
    count_array = np.zeros(n_bins)
    rho = num_mol / box_size**3
    n_sample = 0

    for idx, f in enumerate(file_list):
        print(f'===== dealing with {idx} file: {f}')
        data = torch.load(f)
        qpl = data['qpl_trajectory']
        print('qpl shape', qpl.shape)
        q = qpl[:, 0, pt, :, :]
        n_sample += q.size(0)

        dis_list = []
        for i in range(num_mol):  # loop over O atoms in molecule i
            for j in range(i + 1, num_mol):  # other molecules
                # for k in range(1, 3):   # H atoms in molecule j
                for k in range(1):  # O atom in molecule j
                    disp = q[:, 3 * i, :] - q[:, 3 * j + k, :]  # displacement
                    disp = minimum_image(disp, box_size)
                    dis = torch.norm(disp, dim=-1)  # Euclidean distance
                    dis_list.append(dis.reshape(-1))

        all_d = torch.cat(dis_list, dim=0)
        print("Min distance:", torch.min(all_d).item(), "nm", "Max distance:", torch.max(all_d).item(), "nm")

        # Compute histogram
        counts, bin_edges = np.histogram(all_d.cpu(),
                                         bins=n_bins,
                                         range=(math.floor(r_min), math.ceil(r_max)),
                                         density=False)

        count_array += counts

    grbin = []
    # print(type(count_array), type(bin_edges))
    for c, r in zip(count_array, bin_edges[:-1]):
        # gr = 3 * rho * c * (num_mol - 1) / (4 * math.pi * (r + bin_width) ** 3 - r ** 3)
        shell_volume = (4 / 3) * math.pi * ((r + bin_width) ** 3 - r ** 3)
        gr = 2 * c / ((num_mol - 1) * rho * n_sample * shell_volume)
        grbin.append(gr)
        print(c, r, gr, shell_volume)

    print(type(counts), len(counts), len(bin_edges), len(grbin))
    print('grbin', torch.tensor(grbin).shape, 'edge center', (bin_edges[:-1] + bin_edges[1:]).shape)
    data = {'counts': count_array,
            'gr': torch.tensor(grbin),
            'edge_centers': (bin_edges[:-1] + bin_edges[1:]) / 2}
    torch.save(data, op_name)

    # ---------- Plot ----------
    # Linear scale
    plt.plot(data['edge_centers'], data['gr'])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    num_mol = 8
    box_size = 2.2
    r_min = 0
    r_max = 2
    assert box_size / 2 * 3 ** 0.5 < r_max, 'r max is smaller than diagonal of half box size'
    bin_width = 0.01
    pt = 10000
    # file_list = [f'../../../../Data/LLUF/300k_gap1_nvt_long.pt']
    file_list = [f'../../../../SavedMoldel/LLUF/gap10_b0.01_n128-128-128_d256/0.02_id{i}.pt' for i in range(32)]
    pair_distribution_function(file_list, f'../../../../Data/LLUF/300k_LLUF_histo.pt')
