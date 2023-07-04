import torch


class accumulator:

    def __init__(self, neval):
        self.list_size = 2 * neval
        self.pw_list = None # HK useless code
        self.mb_list = None # HK useless code
        self.clear()

    def accumulate(self, force_id, cur_force, force_type):
        """
        :param force_id: which force or q residual list to accumulate to
        :param cur_force: force or q residual
        :param force_type: either 'pw' or 'mb'
        :return:
        """
        if force_type == 'pw':
            if len(self.pw_list[force_id]) > 0:
                assert self.pw_list[force_id][0].shape[1:] == cur_force.shape[1:], \
                    'New added pw force shape is not the same as the previous ones.'
            self.pw_list[force_id].append(cur_force)
        elif force_type == 'mb':
            # print(len(self.mb_list), force_id)
            if len(self.mb_list[force_id]) > 0:
                # print(self.mb_list[force_id][0].shape, cur_force.shape, force_id, force_type)
                assert self.mb_list[force_id][0].shape[1:] == cur_force.shape[1:], \
                    'New added mb force shape is not the same as the previous ones.'
            self.mb_list[force_id].append(cur_force)

    def clear(self):
        self.pw_list = []
        self.mb_list = []
        for i in range(self.list_size):
            self.pw_list.append([])
            self.mb_list.append([])

    def print(self, e, mode):
        assert len(self.pw_list) == len(self.mb_list), 'Length of pw force list is different from mb force list: {} {}'\
            .format(len(self.pw_list), len(self.mb_list))
        for net_id in range(len(self.pw_list)):
            # print(net_id, len(self.pw_list), len(self.mb_list))
            pw_force = torch.cat(self.pw_list[net_id], dim=0)
            mb_force = torch.cat(self.mb_list[net_id], dim=0)
            assert pw_force.shape == mb_force.shape, 'Shape of pw force is different from mb force shape.'
            print(pw_force.requires_grad, pw_force.shape)

            pw_force = torch.linalg.norm(pw_force, axis=2).flatten()
            mb_force = torch.linalg.norm(mb_force, axis=2).flatten()
            print(pw_force.requires_grad, pw_force.shape)
            if net_id % 2 == 0:
                op_type = 'force'
            else:
                op_type = 'q residual'
            print('PW {} of net {} epoch {} mode {}: sample size {:d} mean {:.4f} std {:.4f} min {:.4f} max {:.4f}'
                  .format(op_type, net_id, e, mode, len(pw_force), torch.mean(pw_force), torch.std(pw_force),
                          torch.min(pw_force), torch.max(pw_force)))

            print('MB {} of net {} epoch {} mode {}: sample size {:d} mean {:.4f} std {:.4f} min {:.4f} max {:.4f}'
                  .format(op_type, net_id, e, mode, len(mb_force), torch.mean(mb_force), torch.std(mb_force),
                          torch.min(mb_force), torch.max(mb_force)))
