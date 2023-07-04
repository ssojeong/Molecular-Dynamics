import json
import torch.optim as optim
from HNN.optimizer import optimizer

class ML_parameters:

    ML_chk_pt_filename      = None            # if None then do not load models

    train_filename          = None
    valid_filename          = None
    write_chk_pt_filename   = None            # filename to save checkpoints
    write_loss_filename     = None            # filename to save loss values
    checkpoint_interval     = None            # adjust this to checkpoint

    train_pts               = None            # selected number of nsamples
    valid_pts               = None            # selected number of nsamples
    seed                    = None

    qp_weight               = None            # loss weight btw q and p
    Lambda                  = None            # Lambda of reqularization term 
    clip_value              = None            # gradient clipping value

    # optimizer parameters
    lr                      = None
    lr_decay_step           = None
    lr_decay_rate           = None
    lr_thrsh                = None
    loss_type               = None
    nepoch                  = None
    batch_size              = None

    # MLP network parameters
    layer_list              = None
    dropout_list            = None
    activation              = None

    on_off_noML             = None

    # fields HNN
    dgrid                   = None
    ngrid                   = None

    #opt_op                  = optim.Adam      # optimizer option
    opt_op                  = optim.SGD        # optimizer option
    opt                     = None             # construct optimizer

    @staticmethod
    def load_dict(json_filename):
        with open(json_filename) as f:
            data = json.load(f)

        ML_parameters.ML_chk_pt_filename        = data['ML_chk_pt_filename']
        ML_parameters.train_filename            = data['train_filename']
        ML_parameters.valid_filename            = data['valid_filename']
        ML_parameters.write_chk_pt_filename     = data['write_chk_pt_filename']
        ML_parameters.write_loss_filename       = data['write_loss_filename']
        ML_parameters.checkpoint_interval       = data['checkpoint_interval']

        ML_parameters.train_pts                 = data['train_pts']
        ML_parameters.valid_pts                 = data['valid_pts']
        ML_parameters.seed                      = data['seed']
        ML_parameters.qp_weight                 = data['qp_weight']
        ML_parameters.Lambda                    = data['Lambda']
        ML_parameters.clip_value                = data['clip_value']
        ML_parameters.lr                        = data['lr']
        ML_parameters.lr_decay_step             = data['lr_decay_step']
        ML_parameters.lr_decay_rate             = data['lr_decay_rate']
        ML_parameters.lr_thrsh                  = data['lr_thrsh']
        ML_parameters.loss_type                 = data['loss_type']        
        ML_parameters.nepoch                    = data['nepoch']
        ML_parameters.batch_size                = data['batch_size']
        ML_parameters.layer_list                = data['layer_list']
        ML_parameters.dropout_list              = data['dropout_list']
        ML_parameters.activation                = data['activation']
        ML_parameters.on_off_noML               = data['on_off_noML']
        ML_parameters.dgrid                     = data['dgrid']
        ML_parameters.ngrid                     = data['ngrid']

        ML_parameters.opt = optimizer(ML_parameters.opt_op, ML_parameters.lr)
