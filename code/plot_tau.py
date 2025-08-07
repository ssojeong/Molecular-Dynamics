# cat pw256mb256lr1e-3 | grep qrmse | grep train | cut -d " " -f 1,2,3,4,5,6,7,8 > pw256mb256lr1e-3qrmse.txt
import sys
import numpy as np
import matplotlib.pyplot as plt
import json

if __name__ == '__main__':
    # python plot_tau.py load_file_tau0.1.dict traj8ws8tau0.1ngrid12w8421ew1repw10poly1l_ \
    # results/traj_len08ws08tau0.1ngrid12api0lw8421ew1repw10poly1l_dpt100

    argv = sys.argv
    load_files = argv[1]
    file_name = argv[2]
    saved_dir = argv[3]

    with open(load_files) as f:
        data = json.load(f)

    print('file name', file_name)

    tau = np.genfromtxt(data['{}traintau'.format(file_name)]);

    tau1 = tau[:, 5];
    tau2 = tau[:, 9];
    tau3 = tau[:, 13]

    epoch = tau[:, 0]

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))

    # ax[0,0].set_title(name,fontsize=10)
    # ax[0].set_title('First trainable param for update q',fontsize=15)
    ax[0].plot(epoch, tau3, 'bo-', color='k', label=r'$\tau_{mb,1}$', zorder=2)
    # ax[1].set_title('Second trainable param for update q',fontsize=15)
    ax[1].plot(epoch, tau2, 'bo-', color='k', label=r'$\tau_{mb,2}$', zorder=2)
    # ax[2].set_title('Trainable param for update p',fontsize=15)
    ax[2].plot(epoch, tau1, 'bo-', color='k', label=r'$\tau_{mb,3}$', zorder=2)

    tau_min = (min(min(tau3), min(tau2), min(tau1)))
    tau_max = (max(max(tau3), max(tau2), max(tau1)))

    for i in range(3):
        ax[i].set_ylim([tau_min - 0.5, tau_max + 0.5])
        ax[i].grid()
        ax[i].set_xlabel('epochs', fontsize=20)
        ax[i].legend(loc='upper right', fontsize=20)
        ax[i].tick_params(axis='x', labelsize=15)
        ax[i].tick_params(axis='y', labelsize=15)

    # fig.suptitle(" # {}".format(discribe), fontsize=15)

    plt.tight_layout()
    # fig.savefig(saved_dir + f'trainable_tau.png', bbox_inches='tight', dpi=200)
    plt.show()
