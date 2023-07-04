import torch


class force_stat:

    def __init__(self, nnet):
        self.list_size = nnet
        self.clear()

    def accumulate(self, force_id, cur_force):
        """
        :param force_id: which force or q residual list to accumulate to
        :param cur_force: force or q residual
        :return:
        """
        if len(self.pw_list[force_id]) > 0:
            assert self.pw_list[force_id][0].shape[1:] == cur_force.shape[1:], \
                'New added pw force shape is not the same as the previous ones.'
        #self.force_list[force_id].append(cur_force) # HK bad code, slow and memory problem
        self.forcex_sum += cur_force.numpy()
        self.forcex_sum_sq += (cur_force*cur_force).numpy()
        self.forcex_min = min(self.forcex_min,torch.min(cur_force).numpy()
        --> the rest of code add accordingly

        self.cntr += 1

    def clear(self):

        self.forcex_sum     = np.zeros((self.list_size)) # make zero
        self.forcex_sq_sum  = --- > change accordingly
        self.forcex_min     = -1e10*np.ones((self.list_size))
        self.forcex_max     =  1e10*np.ones(.....)
        self.forcey_sum     =
        self.forcey_sq_sum  =  
        self.forcey_min     =
        self.forcey_max     = 
        self.cntr = 0

    def print(self, e, mode):
        pass # redo the whole code. 
