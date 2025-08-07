import os
import torch

class checkpoint:
    """
    this class to save and load checkpoints in a dictionary
    """
    def __init__(self, net, opt):
        """
        net_list    : network list for chk pt
        opt         : pass optimizer
        sch         : schedule lr
        sch_reset   : defult : None, otherwise set 'reset' to restart lr that want when retrain
        lr_reset    : float ; start lr that want to give
        """

        self.net = net
        self.opt = opt

        print('checkpoint initialized : net ',net, 'opt ', opt)

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

            if torch.cuda.is_available():
                map_location = lambda storage, loc: storage.cuda()
            else:
                map_location = 'cpu'

            checkpoint = torch.load(full_name, map_location=map_location)
            #print(checkpoint)

            print('Previously trained net state_dict loaded...')
            self.net.load_state_dict(checkpoint['net'])

            print('Previously trained optimizer state_dict loaded...')
            self.opt.load_state_dict(checkpoint['opt'])

            #print('params', self.opt.param_groups[0]["params"])
            #for name, param in self.net.named_parameters():
            #    print(name, ':', param.data)

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

        #assert (self._opt or self._sch is not None), 'opt or sch is None ... so that save error ...'

        full_name = save_filename
        torch.save({'net' : self.net.state_dict(),
                    'opt' : self.opt.state_dict()
                    }, full_name)


