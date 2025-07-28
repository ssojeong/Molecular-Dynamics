import numpy as np

class force_stat:

    def __init__(self):
        self.dim = 2
        self.clear()

    def accumulate(self, cur_force):
        """
        :param force_id: which force or q residual list to accumulate to
        :param cur_force: force or q residual
        :return:
        """
        # cur_force shape=[nsample,nparticle,dim]
        cur_force = cur_force.view(-1, 2).clone().detach().cpu().numpy() # -1 :  batch * nparticles
        self.force_mean   += np.mean(cur_force,axis=0)
        self.force_stdd   += np.std(cur_force,axis=0)
        self.force_min[0]  = min(self.force_min[0], np.min(cur_force,axis=0)[0]) # min x
        self.force_max[0]  = max(self.force_max[0], np.max(cur_force,axis=0)[0]) # max x
        self.force_min[1]  = min(self.force_min[1], np.min(cur_force, axis=0)[1]) # min y
        self.force_max[1]  = max(self.force_max[1], np.max(cur_force,axis=0)[1]) # max y

        self.cntr += 1

    def clear(self):
        self.force_mean  = np.zeros(self.dim) # make zero
        self.force_stdd  = np.zeros(self.dim)
        self.force_min   =  1e10*np.ones(self.dim)
        self.force_max   = -1e10*np.ones(self.dim)
        self.cntr = 0

    def print(self, e, label):
        print(e, label, 'update function  cntr ' , self.cntr)

        if self.cntr > 0:

            mean_x = self.force_mean[0] / self.cntr
            stdd_x = self.force_stdd[0] / self.cntr

            mean_y = self.force_mean[1] / self.cntr
            stdd_y = self.force_stdd[1] / self.cntr

            print('For x axis: mean {:.4f} std {:.4f} min {:.4f} max {:.4f}'
                  .format(mean_x, stdd_x, self.force_min[0], self.force_max[0]))
            print('For y axis: mean {:.4f} std {:.4f} min {:.4f} max {:.4f}'
                  .format(mean_y, stdd_y, self.force_min[1], self.force_max[1]))
        else:
            print(e,label,' force_stat.py, no stat to print')


