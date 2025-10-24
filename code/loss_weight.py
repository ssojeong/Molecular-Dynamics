import torch
import numpy as np
import sys
import json
# import matplotlib
# matplotlib.use('Agg')
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

    print(data)

    tqrmse = np.genfromtxt(data[f'{name}trainqrmse'])
    vqrmse = np.genfromtxt(data[f'{name}validqrmse'])
    tprmse = np.genfromtxt(data[f'{name}trainprmse'])
    vprmse = np.genfromtxt(data[f'{name}validprmse'])

    tlr = tqrmse[:, 1]
    lr = tqrmse[:, 3]
    ttqrmse = tqrmse[:, 1]
    vtqrmse = vqrmse[:, 1]

    qrmse_t = tqrmse[:, 5:]
    qrmse_v = vqrmse[:, 5:]

    print('qrmse_t', qrmse_t.shape, qrmse_t[0, :])
    prmse_t = tprmse[:, 5:]
    prmse_v = vprmse[:, 5:]  # mode train

    tepoch = ttqrmse
    vepoch = vtqrmse

    # plt.ion()
    fig, ax = plt.subplots(nrows=ws, ncols=2, figsize=(10, 8))  # Increased height for multiple rows
    # Handle the case when ws=1 (ax becomes 1D instead of 2D)
    if ws == 1:
        ax = np.array([ax])
    print('ws', ws, 'ax', ax.shape)
    for i in range(ws):
        # Use i to index the appropriate column in your data arrays
        ax[i, 0].plot(tepoch, qrmse_t[:, i], 'bo-', label='train', zorder=2)
        ax[i, 0].plot(vepoch, qrmse_v[:, i], 'o-', label='valid', c='orange', zorder=1)
        ax[i, 1].plot(tepoch, prmse_t[:, i], 'bo-', label='train', zorder=2)
        ax[i, 1].plot(vepoch, prmse_v[:, i], 'o-', label='valid', c='orange', zorder=1)

        # Set labels and titles for each row
        ax[i, 0].set_ylabel(f'$L_{i+1}$', fontsize=15)

        # Apply grid and legend to both subplots in this row
        for j in range(2):
            ax[i, j].grid()
            ax[i, j].legend(loc='upper right', fontsize=10)
            ax[-1, j].set_xlabel('epochs', fontsize=15)

    ax[0, 0].set_title(f'q L2 norm', fontsize=12)
    ax[0, 1].set_title(f'p L2 norm', fontsize=12)
    fig.suptitle(f"{title}", fontsize=12)
    plt.tight_layout()

    print("About to display plot...")
    plt.show()
    print("Plot display completed")

