import sys
import torch
import matplotlib.pyplot as plt

if __name__ == '__main__':

    argv = sys.argv

    if len(argv) != 3:
        print('usage <programe> <file1> <file2>')
        quit()

    file1 = argv[1]
    file2 = argv[2]

    data1 = torch.load(file1)
    data2 = torch.load(file2)

    qp1 = data1['qp_trajectory']
    qp2 = data2['qp_trajectory']

    init_qp1 = qp1[:,:,0,:,:]
    qp1_strike_append = qp1[:,:,1,:,:]

    init_qp2 = qp2[:,:,0,:,:]
    qp2_strike_append = qp2[:,:,1,:,:]

    init_p1 = init_qp1[:,1,:,:]
    p1_strike_append = qp1_strike_append[:,1,:,:]

    init_p2 = init_qp2[:,1,:,:]
    p2_strike_append = qp2_strike_append[:,1,:,:]

    init_p1 = init_p1.reshape(-1)
    p1_strike_append = p1_strike_append.reshape(-1)

    init_p2 = init_p2.reshape(-1)
    p2_strike_append = p2_strike_append.reshape(-1)


    plt.title('no. of original data {} crash data {}'.format(qp1.shape[0], qp2.shape[0]))
    plt.hist(init_p1.detach().numpy(), bins=100, alpha=.5, label='init p from original data')
    plt.hist(init_p2.detach().numpy(), bins=100, alpha=.5, label='init p from crash data')
    plt.xlabel('init p', fontsize=20)
    plt.ylabel('hist', fontsize=20)
    plt.legend()
    plt.show()

    plt.title('no. of original data {} crash data {}'.format(qp1.shape[0], qp2.shape[0]))
    plt.hist(p1_strike_append.detach().numpy(), bins=100, alpha=.5, label='p strike append from original data')
    plt.hist(p2_strike_append.detach().numpy(), bins=100, alpha=.5, label='p strike append from crash data')
    plt.xlabel('p strike append', fontsize=20)
    plt.ylabel('hist', fontsize=20)
    plt.legend()
    plt.show()
