import torch
import matplotlib.pyplot as plt
import numpy as np


data = torch.load('/home/liuwei/Projects/LLUF/test.pt')
traj = data["qp"]
print(type(traj), traj.shape)
# dis = torch.norm(traj[:, :, 0, :3] - traj[:, :, 1, :3], dim=-1)
# print(torch.min(dis), torch.mean(dis.reshape(-1)), torch.std(dis.reshape(-1)))
#
print(torch.min(traj[:, :, :, :, 0]), torch.max(traj[:, :, :, :, 0]))

min_dis_list = []
dis_list = []
for i in range(8):
    for j in range(i+1, 8):
        for k in range(1, 3):
            # print(3*i, 3*j+k)
            dis = torch.norm(traj[:, :, 3*i, :, 0] - traj[:, :, 3*j+k, :, 0], dim=-1)

            # Flatten to get global min
            dis_flat = dis.reshape(-1)
            dis_list.append(dis_flat)
            min_val, min_idx = torch.min(dis_flat, dim=0)

            # Convert flat index back to (traj, frame) coordinates
            n_traj, n_frame = dis.shape
            # print(dis.shape, min_idx)
            traj_idx = min_idx // n_frame
            frame_idx = min_idx % n_frame

            # min_dis_list.append(min_val.item())
            # print(3*i, 3*j+k, (traj_idx.item(), frame_idx.item()), min_val.item())
            # min_dis_list.append(torch.min(dis))

all_d = torch.cat(dis_list, dim=0)
print(torch.min(all_d))
# max_d = (3**0.5) * (2.2 / 2)  # ~1.905 nm
# bin_width = 0.01
# num_bins = int(max_d / bin_width)
#
# # Compute histogram
# counts, edges = torch.histogram(all_d, bins=num_bins, range=(0.0, max_d))
# bin_centers = (edges[:-1] + edges[1:]) / 2
# # print(min(min_dis_list))
# # o = traj[166, 141, 0, :, 0] + 1.1
# # h = traj[166, 141, 17, :, 0] + 1.1
# # print(o, h, torch.norm(o-h))
# # print(data["traj_id"][166], data["times"][141])
# # print(traj[0, 0, :, :])
#
# plt.figure(figsize=(8, 5))
# plt.bar(bin_centers.numpy(), counts.numpy(), width=bin_width, align='center', edgecolor='k')
# plt.xlabel("Distance (nm)")
# plt.ylabel("Counts")
# plt.title("O···H Distance Histogram")
# # Dense X-axis ticks (every 0.1 nm)
# x_ticks = np.arange(0, max_d + 0.01, 0.1)
# plt.xticks(x_ticks)
#
# # Grid with more vertical lines
# plt.grid(True, which='both', axis='both', linestyle='--', alpha=0.7)
#
# plt.tight_layout()
# plt.show()
