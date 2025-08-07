import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
import json

if __name__ == '__main__' :

    argv       = sys.argv
    load_files = argv[1]
    name = argv[2]
    loss_weights = argv[3].split(',')
    title = argv[4]

    with open(load_files) as f:
       data = json.load(f)

    tqrmse = np.genfromtxt(data['{}trainqrmse'.format(name)]); vqrmse = np.genfromtxt(data['{}validqrmse'.format(name)])
    tqshape = np.genfromtxt(data['{}trainqshape'.format(name)]); vqshape = np.genfromtxt(data['{}validqshape'.format(name)])
    tprmse = np.genfromtxt(data['{}trainprmse'.format(name)]); vprmse = np.genfromtxt(data['{}validprmse'.format(name)])
    tpshape = np.genfromtxt(data['{}trainpshape'.format(name)]); vpshape = np.genfromtxt(data['{}validpshape'.format(name)])
    termse = np.genfromtxt(data['{}trainermse'.format(name)]); vermse = np.genfromtxt(data['{}validermse'.format(name)])
    teshape = np.genfromtxt(data['{}traineshape'.format(name)]); veshape = np.genfromtxt(data['{}valideshape'.format(name)])
    trelureg = np.genfromtxt(data['{}trainrelurep'.format(name)]); vrelureg = np.genfromtxt(data['{}validrelurep'.format(name)])
    poly = np.genfromtxt(data['{}trainpoly'.format(name)])
    trep = np.genfromtxt(data['{}trainrep'.format(name)]); vrep = np.genfromtxt(data['{}validrep'.format(name)])
    ttotal = np.genfromtxt(data['{}train'.format(name)]); vtotal = np.genfromtxt(data['{}valid'.format(name)])

    for i in range(len(loss_weights)):
        if isinstance(loss_weights[i], float):
            loss_weights[i] = float(loss_weights[i])
        else:
            loss_weights[i] = eval(loss_weights[i])

    #c = [i for i in loss_weights if i !=0]

    tlr = tqrmse[:,1]; lr = tqrmse[:,3]
    ttqrmse = tqrmse[:, 1]; vtqrmse =  vqrmse[:, 1]

    qrmse_t = tqrmse[:, 5:]; qrmse_v = vqrmse[:, 5:]

    qshape_t = tqshape[:, 5:]; qshape_v = vqshape[:, 5:]  # mode train

    prmse_t = tprmse[:, 5:]; prmse_v = vprmse[:, 5:]  # mode train
    pshape_t = tpshape[:, 5:]; pshape_v = vpshape[:, 5:]  # mode train

    ermse_t = termse[:, 5:]; ermse_v = vermse[:, 5:]  # mode train
    eshape_t = teshape[:, 5:]; eshape_v = veshape[:,5:]  # mode train

    relureg_t = trelureg[:, 5:]; relureg_v = vrelureg[:, 5:]  # mode train

    rep_t = trep[:, 1:]; rep_v = vrep[:, 1:]  # mode train
    poly_t = poly[:, 1]; poly_v = poly[:, 1]  # mode train

    loss_t = ttotal[:, 5:]; loss_v = vtotal[:, 5:]  # mode train

    tepoch = ttqrmse
    vepoch = vtqrmse

    fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(12, 10))

    for i in range(4):
        ax[i, 0].plot(tepoch, qrmse_t[:,2*i+1], 'bo-', label='train', zorder=2)
        ax[i, 0].plot(vepoch, qrmse_v[:,2*i+1], 'o-', label='valid', c='orange', zorder=1)
        ax[i, 1].plot(tepoch, prmse_t[:,2*i+1], 'bo-', label='train', zorder=2)
        ax[i, 1].plot(vepoch, prmse_v[:,2*i+1], 'o-', label='valid', c='orange', zorder=1)
        ax[i, 2].plot(tepoch, ermse_t[:,2*i+1], 'bo-', label='train', zorder=2)
        ax[i, 2].plot(vepoch, ermse_v[:,2*i+1], 'o-', label='valid', c='orange', zorder=1)
        # ax[i, 2].set_ylim([-0.05, 0.5])
        ax[i, 2].set_ylim([-0.05, 100000])
        ax[i, 3].plot(tepoch, relureg_t[:,2*i+1], 'bo-', label='train', zorder=2)
        ax[i, 3].plot(vepoch, relureg_v[:,2*i+1], 'o-', label='valid', c='orange', zorder=1)
        #ax[i, 3].set_ylim([-0.001, 10])
        ax[i, 0].set_ylabel(r'$L_{}$'.format(2*i+2), fontsize=15)

    ax[0, 0].set_title('q L2 norm', fontsize=15)
    ax[0, 1].set_title('p L2 norm', fontsize=15)
    ax[0, 2].set_title('e L2 norm', fontsize=15)
    ax[0, 3].set_title('rep ReLU', fontsize=15)

    for i in range(4):
        for j in range(4):
            ax[i, j].grid()
            ax[i, j].legend(loc='upper right', fontsize=10)

    for i in range(4):
        ax[3, i].set_xlabel('epochs', fontsize=15)

    fig.suptitle("# {}".format(title), fontsize=15)
    plt.tight_layout()
    plt.show()
    plt.close()

