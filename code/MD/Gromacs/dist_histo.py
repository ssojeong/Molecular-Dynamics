import torch
import matplotlib.pyplot as plt
import numpy as np

# ---------- Parameters ----------
box_length = 2.2  # nm, cubic box size
bin_width = 0.01
max_d = (3**0.5) * (box_length / 2)  # maximum possible distance
num_bins = int(max_d / bin_width)

# ---------- Load Data ----------
data = torch.load('/home/liuwei/Projects/LLUF/300k.pt')
traj = data["qp"]  # shape: [traj_id, frame, atom, coord, type]
print("Trajectory shape:", traj.shape)

# ---------- Apply PBC ----------
def minimum_image(vec, box):
    """Apply minimum image convention to displacement vectors."""
    return vec - box * torch.round(vec / box)

# ---------- Distance Calculation ----------
dis_list = []
for i in range(8):          # loop over O atoms in molecule i
    for j in range(i + 1, 8):   # other molecules
        for k in range(1, 3):   # H atoms in molecule j
            disp = traj[:, :, 3*i, :, 0] - traj[:, :, 3*j + k, :, 0]  # displacement
            disp = minimum_image(disp, box_length)
            dis = torch.norm(disp, dim=-1)  # Euclidean distance
            dis_list.append(dis.reshape(-1))

all_d = torch.cat(dis_list, dim=0)
print("Min distance:", torch.min(all_d).item(), "nm")

# ---------- Histogram (occurrence) ----------
counts, edges = torch.histogram(all_d, bins=num_bins, range=(0.0, max_d))
counts = counts.numpy()
edges = edges.numpy()
bin_centers = (edges[:-1] + edges[1:]) / 2

# Convert counts → occurrence (normalized)
occurrence = counts / counts.sum()

# ---------- Plot ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# Linear scale
ax1.bar(bin_centers, occurrence, width=bin_width, align='center', edgecolor='k')
ax1.set_ylabel("Occurrence")
ax1.set_title("O···H Distance Histogram (Linear Y)")

# Log scale
ax2.bar(bin_centers, occurrence, width=bin_width, align='center', edgecolor='k')
ax2.set_xlabel("Distance (nm)")
ax2.set_ylabel("Occurrence")
ax2.set_yscale('log')
ax2.set_title("O···H Distance Histogram (Log Y)")

# Dense x-axis ticks
x_ticks = np.arange(0, max_d + 0.01, 0.1)
ax2.set_xticks(x_ticks)

# Grid
for ax in [ax1, ax2]:
    ax.grid(True, which='both', axis='both', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
