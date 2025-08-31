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

    for idx, f in enumerate(file_list):
        print(f'===== dealing with {idx} file: {f}')
        data = torch.load(f)
        qpl = data['qpl_trajectory']
        print('qpl shape', qpl.shape)
        q = qpl[:, 0, pt, :, :]
        # print(q.shape)

        dis_list = []
        for i in range(8):  # loop over O atoms in molecule i
            for j in range(i + 1, 8):  # other molecules
                # for k in range(1, 3):   # H atoms in molecule j
                for k in range(1):  # O atom in molecule j
                    disp = q[:, 3 * i, :] - q[:, 3 * j + k, :]  # displacement
                    disp = minimum_image(disp, box_size)
                    dis = torch.norm(disp, dim=-1)  # Euclidean distance
                    dis_list.append(dis.reshape(-1))

        all_d = torch.cat(dis_list, dim=0)
        print("Min distance:", torch.min(all_d).item(), "nm", "Max distance:", torch.max(all_d).item(), "nm")

        # Compute histogram
        counts, bin_edges = np.histogram(all_d,
                                         bins=n_bins,
                                         range=(math.floor(r_min), math.ceil(r_max)),
                                         density=False)

        print(type(counts), len(counts), len(bin_edges))
        count_array += counts

    torch.save({'counts': count_array, 'edge_centers': (bin_edges[:-1] + bin_edges[1:]) / 2}, op_name)

    # # Convert counts → occurrence (normalized)
    # occurrence = count_array / count_array.sum()
    # edges = bin_edges
    # bin_centers = (edges[:-1] + edges[1:]) / 2
    #
    # # ---------- Plot ----------
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    #
    # # Linear scale
    # ax1.bar(bin_centers, occurrence, width=bin_width, align='center', edgecolor='k')
    # ax1.set_ylabel("Occurrence")
    # ax1.set_title("O···H Distance Histogram (Linear Y)")
    #
    # # Log scale
    # ax2.bar(bin_centers, occurrence, width=bin_width, align='center', edgecolor='k')
    # ax2.set_xlabel("Distance (nm)")
    # ax2.set_ylabel("Occurrence")
    # ax2.set_yscale('log')
    # ax2.set_title("O···H Distance Histogram (Log Y)")
    #
    # # Dense x-axis ticks
    # x_ticks = np.arange(0, r_max + 0.01, 0.1)
    # ax2.set_xticks(x_ticks)
    #
    # # Grid
    # for ax in [ax1, ax2]:
    #     ax.grid(True, which='both', axis='both', linestyle='--', alpha=0.7)
    #
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    box_size = 2.2
    r_min = 0
    r_max = box_size / 2 * 3 ** 0.5
    bin_width = 0.01
    pt = -1
    file_list = [f'../../../../Data/LLUF/300k_gap1_nvt_long.pt']
    pair_distribution_function(file_list, f'../../../../Data/LLUF/300k_gromacs_histo.pt')
