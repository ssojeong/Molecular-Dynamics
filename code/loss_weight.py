import torch
import numpy as np
import sys
import json
import matplotlib.pyplot as plt

if __name__ == '__main__':
    argv = sys.argv
    print(argv)
    load_files = argv[1]
    name = argv[2]
    title = argv[3]
    ws = int(argv[4])

    print(load_files, name, title)

    with open(load_files) as f:
        data = json.load(f)

    tqrmse = np.genfromtxt(data[f'{name}trainqrmse'])
    vqrmse = np.genfromtxt(data[f'{name}validqrmse'])
    tprmse = np.genfromtxt(data[f'{name}trainprmse'])
    vprmse = np.genfromtxt(data[f'{name}validprmse'])

    tepoch = tqrmse[:, 1]
    vepoch = vqrmse[:, 1]
    qrmse_t = tqrmse[:, 5:]
    qrmse_v = vqrmse[:, 5:]
    prmse_t = tprmse[:, 5:]
    prmse_v = vprmse[:, 5:]

    # --- Create subplots ---
    fig, ax = plt.subplots(nrows=ws, ncols=2, figsize=(10, 8), sharex='col')

    # Handle case when ws = 1
    if ws == 1:
        ax = np.expand_dims(ax, axis=0)

    # --- Plot each subplot ---
    for i in range(ws):
        # Left column: qRMSE
        ax[i, 0].plot(tepoch, qrmse_t[:, i], 'b-', label='Train', zorder=2)
        ax[i, 0].plot(vepoch, qrmse_v[:, i], '-', color='orange', label='Valid', zorder=1)

        # Right column: pRMSE
        ax[i, 1].plot(tepoch, prmse_t[:, i], 'b-', label='Train', zorder=2)
        ax[i, 1].plot(vepoch, prmse_v[:, i], '-', color='orange', label='Valid', zorder=1)

        # Y-labels per row
        ax[i, 0].set_ylabel(f'$L_{{{i+1}}}$', fontsize=13)

        # Grid for all
        for j in range(2):
            ax[i, j].grid(alpha=0.3)

    # --- Shared x-labels per column ---
    for j in range(2):
        ax[-1, j].set_xlabel('Epochs', fontsize=14)

    # --- Titles per column ---
    ax[0, 0].set_title('q L2 norm', fontsize=14)
    ax[0, 1].set_title('p L2 norm', fontsize=14)

    # --- Legend only in the top-left subplot ---
    ax[0, 0].legend(frameon=False, fontsize=12, loc='upper right')

    # --- Overall title ---
    fig.suptitle(title, fontsize=14, y=0.95)

    # plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
    plt.show()
