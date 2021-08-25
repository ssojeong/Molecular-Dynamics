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

    param1 = torch.load(file1)
    w = param1['weights']
    b = param1['bias']

    param2 = torch.load(file2)
    grad_w = param2['weights']
    grad_b = param2['bias']

    plt.title('histogram of weights')
    plt.hist(w.detach().numpy(), bins=100, alpha=.5, label='weight')
    plt.xlabel('weight', fontsize=20)
    plt.ylabel('hist', fontsize=20)
    plt.legend()
    plt.show()

    plt.title('histogram of bias')
    plt.hist(b.detach().numpy(), bins=100, alpha=.5, label='bias')
    plt.xlabel('bias', fontsize=20)
    plt.ylabel('hist', fontsize=20)
    plt.legend()
    plt.show()

    plt.title('histogram of weights gradient')
    plt.hist(grad_w.numpy(), bins=100, alpha=.5, label='weight.grad')
    plt.xlabel('weight.grad', fontsize=20)
    plt.ylabel('hist', fontsize=20)
    plt.legend()
    plt.show()

    plt.title('histogram of bias gradient')
    plt.hist(grad_b.numpy(), bins=100, alpha=.5, label='bias.grad')
    plt.xlabel('bias.grad', fontsize=20)
    plt.ylabel('hist', fontsize=20)
    plt.legend()
    plt.show()