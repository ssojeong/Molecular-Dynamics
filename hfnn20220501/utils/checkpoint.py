import os
import torch

class checkpoint:
    """
    this class to save and load checkpoints in a dictionary
    """
    def __init__(self, net_list, opt = None, opt2 = None, sch = None):
        """
        net_list    : network list for chk pt
        opt         : pass optimizer
        sch         : schedule lr
        sch_reset   : defult : None, otherwise set 'reset' to restart lr that want when retrain
        lr_reset    : float ; start lr that want to give
        """

        self.net_list = net_list
        self._opt = opt
        self._opt2 = opt2
        self._sch = sch
        self._sch_reset = None # test
        self.lr_reset = None  # test
        print('checkpoint initialized : net list ',net_list,' opt ',opt,' opt2 ',opt2, 'sch', sch)

    # ===================================================
    def load_checkpoint(self, load_filename):

        ''' function to load saved models
            remember to first initialize the models and optimizer, then load the dictionary

        Parameters
        ----------
        load_filename : string
                load saved models. if not, quit
        '''

        full_name = load_filename

        if os.path.isfile(full_name):

            print("=> loading checkpoint '{}'".format(full_name))
            checkpoint = torch.load(full_name)
            print(checkpoint)

            # load models weights state_dict
            for net_id, nlist in enumerate(checkpoint['net_list']):
                self.net_list[net_id].load_state_dict(nlist.state_dict())
                print('Previously net_list state_dict loaded...')

            if self._opt is not None:
                self._opt.load_state_dict(checkpoint['optimizer'])
                print('Previously trained optimizer state_dict loaded...')

            if self._opt2 is not None:
                self._opt2.load_state_dict(checkpoint['optimizer2'])
                print('Previously trained optimizer2 state_dict loaded...')

            if self._sch is not None:
                if self._sch_reset:
                    print('reset sch lr ....',self.lr_reset)
                    self._opt.param_groups[0]['lr'] = self.lr_reset
                else:
                    self._sch.load_state_dict(checkpoint['scheduler'])
                    print('Previously trained scheduler state_dict loaded...')

        else:
            print("=> no checkpoint found at '{}'".format(full_name))
            quit()

    # ===================================================
    def save_checkpoint(self, save_filename):

        ''' function to record the state after each training

        Parameters
        ----------
        save_filename : string
        '''

        assert (self._opt or self._sch is not None), 'opt or sch is None ... so that save error ...'

        full_name = save_filename
        torch.save({'net_list'  : self.net_list,
                    'optimizer' : self._opt.state_dict(),
                    'optimizer2': self._opt2.state_dict(),
                    'scheduler' : self._sch.state_dict()}, full_name)


