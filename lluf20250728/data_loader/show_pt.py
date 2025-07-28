import sys
import torch

if __name__ == '__main__':
    # make sure that multiple components correct or not
    argv = sys.argv
    filename = argv[1]

    data = torch.load(filename)

    qpl_trajectory = data['qpl_trajectory']
    tau_short     = data['tau_short']
    tau_long      = data['tau_long']

    print('qpl shape ', qpl_trajectory.shape, 'tau short ', tau_short, 'tau long ', tau_long)
  

