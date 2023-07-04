import numpy as np
import torch

class force:
    def __init__(self):
        self.dim = 2
        self.clear()

    def accumulate(self, force_id, cur_force, tau):
        """
        :param force_id: which force or q residual list to accumulate to
        :param cur_force: force or q residual
        :return:
        """
        print('save force ... filename:  f{}_s{}_tau{}.pt'.format(force_id,cur_force.shape[0],tau))
        torch.save(cur_force, 'f{}_s{}_tau{}.pt'.format(force_id,cur_force.shape[0],tau))
        cur_force = cur_force.view(-1, 2).clone().detach().cpu().numpy()  # -1 :  nsamples * nparticles
        self.force_mean = np.mean(cur_force, axis=0)
        self.force_stdd = np.std(cur_force, axis=0)
        self.force_min[0] = np.min(cur_force, axis=0)[0]  # min x
        self.force_max[0] = np.max(cur_force, axis=0)[0]  # max x
        self.force_min[1] = np.min(cur_force, axis=0)[1]  # min y
        self.force_max[1] = np.max(cur_force, axis=0)[1]  # max y


    def clear(self):
        self.force_mean = np.zeros((self.dim))  # make zero
        self.force_stdd = np.zeros((self.dim))
        self.force_min = 1e10 * np.ones((self.dim))
        self.force_max = -1e10 * np.ones((self.dim))

    def print(self):
        print('force stat .... ' )

        mean_x = self.force_mean[0]
        stdd_x = self.force_stdd[0]

        mean_y = self.force_mean[1]
        stdd_y = self.force_stdd[1]

        print('For x axis: mean {:.4f} std {:.4f} min {:.4f} max {:.4f}'
                .format(mean_x, stdd_x, self.force_min[0], self.force_max[0]))
        print('For y axis: mean {:.4f} std {:.4f} min {:.4f} max {:.4f}'
                .format(mean_y, stdd_y, self.force_min[1], self.force_max[1]))
