import sys
import torch
import itertools
import matplotlib.pyplot as plt

if __name__ == '__main__':

    argv = sys.argv
    
    if len(argv) != 8:
        print('usage <programe> <file1> <ts1> <name1> <file2> <ts2> <name2> <temp>')
        quit()

    file1 = argv[1]
    ts1   = float(argv[2])
    name1 = argv[3]
    file2 = argv[4]
    ts2   = float(argv[5])
    name2 = argv[6]
    temp  = argv[7]

    crash_info1 = torch.load(file1)
    crash_iter1 = crash_info1['crash_niter']
    crash_time_step1 = crash_iter1*ts1 + ts1 
    crash_ct1 = crash_info1['crash_nct']
    accum1 = itertools.accumulate(crash_ct1)
    print(crash_iter1.shape, crash_ct1.shape, torch.sum(crash_ct1))

   
    crash_info2 = torch.load(file2)
    crash_iter2 = crash_info2['crash_niter']
    crash_time_step2 = crash_iter2*ts2 + ts2
    crash_ct2 = crash_info2['crash_nct']
    accum2 = itertools.accumulate(crash_ct2)
    print(crash_iter2.shape, crash_ct2.shape, torch.sum(crash_ct2))
        
    plt.title('histogram of interation at crash given at T={}'.format(temp),fontsize=15)
    # plt.bar(crash_iter1, list(accum1), alpha=.5, label = name1)
    # plt.bar(crash_iter2, list(accum2), alpha=.5, label = name2)
    plt.plot(crash_time_step1, list(accum1), '*', alpha=.5, label = name1)
    plt.plot(crash_time_step2, list(accum2), 'x', alpha=.5, label = name2)
    plt.xlabel('time',fontsize=20)
    plt.ylabel('accumulate function',fontsize=20)
    plt.legend()
    plt.grid()
    plt.show()
