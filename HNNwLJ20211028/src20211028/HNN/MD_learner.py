from HNN.loss                 import qp_MSE_loss
from HNN.loss                 import qp_MAE_loss
from HNN.loss                 import qp_exp_loss
from HNN.checkpoint           import checkpoint
import torch.nn as nn
import time
import math

import torch

class MD_learner:

    ''' MD_learner class to help train, validate, retrain, and save '''

    _verbose_interval = 5     # adjust this to show results
    _obj_count = 0

    def __init__(self, linear_integrator_obj, 
                 any_HNN_obj, phase_space, opt, 
                 sch, data_loader, pothrsh, qp_weight, Lambda, clip_value, lr_thrsh,loss_type, system_logs, load_model_file=None):
        '''
        Parameters
        ----------
        linear_integrator_obj : use for integrator using large time step
        any_HNN_obj : pass any HNN object to this container
        phase_space : contains q_list, p_list as input
                    q list shape is [nsamples, nparticle, DIM]
        opt         : create one optimizer from two models parameters
        sch         : lr decay 0.99 every 100 epochs
        data_loader : DataLoaders on Custom Datasets
                 two tensors contain train and valid data
                 each shape is [nsamples, 2, niter, nparticle, DIM] , here 2 is (q,p)
                 niter is initial and append strike iter so that 2
        load_model_file : file for save or load them
                 default is None
        '''

        MD_learner._obj_count += 1
        assert (MD_learner._obj_count == 1), type(self).__name__ + " has more than one object"

        self.linear_integrator = linear_integrator_obj
        self.any_HNN = any_HNN_obj
        self.data_loader = data_loader
        self.pothrsh = pothrsh
        self.Lambda  = Lambda

        self._phase_space = phase_space
        self._opt = opt
        self._sch = sch
        self.chk_pt = checkpoint(self.any_HNN.get_netlist(), self._opt, self._sch)
        self.lr_thrsh = lr_thrsh

        self.system_logs = system_logs

        if load_model_file is not None: self.chk_pt.load_checkpoint(load_model_file)

        self.w          = qp_weight

        if loss_type == 'MSE_loss':
            print('loss type : mse_loss')
            self._loss      = qp_MSE_loss
        elif loss_type == 'MAE_loss':
            print('loss type : mae_loss')
            self._loss      = qp_MAE_loss
        elif loss_type == 'exp_loss':
            print('loss type : exp_loss')
            self._loss = qp_exp_loss
        else:
            assert (False), 'invalid loss type given'

        self.tau_cur = self.data_loader.data_set.train_set.data_tau_long
        boxsize      = self.data_loader.data_set.train_set.data_boxsize

        self.clip_value = clip_value

        self._phase_space.set_l_list(boxsize)

        print('MD_learner initialized : tau_cur ',self.tau_cur, 'pothrsh', pothrsh, 'Lambda', self.Lambda, 'clip value', self.clip_value)

    # ===================================================
    def train_one_epoch(self):
        ''' function to train one epoch'''

        self.any_HNN.train()
        train_loss  = 0.
        train_qloss = 0.
        train_ploss = 0.
        train_regloss = 0.

        for step, (input, label) in enumerate(self.data_loader.train_loader):

            #assert ( (torch.isnan(input).any() and torch.isnan(label).any()) == False ), 'input or label get nan......'

            self._opt.zero_grad()

            #print('=======startd zero grad ================================= ')
            #self.any_HNN.print_grad()
            #print('=======end zero grad ================================= ')
            # clear out the gradients of all variables in this optimizer (i.e. w,b)

            # input shape, [nsamples, (q,p,boxsize)=3, nparticle, DIM]
            self._phase_space.set_q(input[:, 0, :, :])
            self._phase_space.set_p(input[:, 1, :, :])
            self._phase_space.set_l_list(input[:, 2, :, :])

            qpl_list, crash_idx = self.linear_integrator.one_step(self.any_HNN, self._phase_space, self.tau_cur)
            # qp_list shape, [nsamples, (q,p,boxsize)=3, nparticle, DIM]

            q_predict = qpl_list[:, 0, :, :]; p_predict = qpl_list[:, 1, :, :]
            # q_predict shape, [nsamples, nparticle, DIM]

            q_label   = label[:, 0, :, :]  ; p_label   = label[:, 1, :, :]
            # q_label shape, [nsamples, nparticle, DIM]

            if __debug__:  # any elements have nan then become true ,so that get error
                assert ((input[:, 2, :, :] == label[:, 2, :, :]).all() == True), 'error .. not match boxsize ...'

            train_predict = (q_predict, p_predict)
            train_label = (q_label, p_label)

            qploss,qloss,ploss = self._loss(train_predict, train_label, self._phase_space, self.w)

            # regularization - tune the function by adding an additional penalty term in loss ; pothrsh=0.7x
            m = nn.ReLU()
            reg_loss = m(self.any_HNN.potential_rep(q_predict, self._phase_space) - self.pothrsh)

            loss = qploss + self.Lambda * reg_loss

            # only open to check save param min-max to plot distribution
            # self.any_HNN.check_param_minmax()

            loss.backward()
            # backward pass : compute gradient of the loss wrt models parameters

            # only open to check save param grad min-max to plot distribution
            # self.any_HNN.check_grad_minmax()
            # quit()

            # Gradient value clipping
            # comment when do param max-min
            for n in self.any_HNN.get_netlist():
                nn.utils.clip_grad_value_(n.parameters(),clip_value=self.clip_value)

            # for n in self.any_HNN.get_netlist():
            #    for name, param in n.named_parameters():
            #        #print("Model Parameters", name, torch.isfinite(param.grad).all())
            #        print("Model Parameters", name, param.grad)
            #        assert (torch.isfinite(param.grad).all() == True), 'param grad get nan .....'

            self._opt.step()
            # self.any_HNN.print_grad()
            # if step==2: quit()

            train_loss  += loss.item()   # get the scalar output
            train_qloss += qloss.item()  # get the scalar output
            train_ploss += ploss.item()  # get the scalar output
            train_regloss  += reg_loss.item() # get the scalar output

        return train_loss / (step+1), train_qloss / (step+1), train_ploss / (step+1), train_regloss / (step+1)

    # ===================================================
    def valid_one_epoch(self):
        ''' function to valid one epoch'''

        self.any_HNN.eval()
        valid_loss  = 0.
        valid_qloss = 0.
        valid_ploss = 0.
        self.pred_valid_time  = 0.

        for step, (input, label) in enumerate(self.data_loader.val_loader):

            #assert ( (torch.isnan(input).all() and torch.isnan(label).all()) == False ), 'input or label get nan......'

            # input shape, [nsamples, (q,p)=2, nparticle, DIM]
            self._phase_space.set_q(input[:, 0, :, :])
            self._phase_space.set_p(input[:, 1, :, :])
            self._phase_space.set_l_list(input[:, 2, :, :])

            start_pred = time.time()
            qpl_list, crash_idx = self.linear_integrator.one_step(self.any_HNN, self._phase_space, self.tau_cur)
            # qp_list shape, [nsamples, (q,p)=2, nparticle, DIM]
            end_pred = time.time()

            q_predict = qpl_list[:, 0, :, :]; p_predict = qpl_list[:, 1, :, :] # q_predict shape, [nsamples, nparticle, DIM]

            q_label   = label[:, 0, :, :]  ; p_label   = label[:, 1, :, :]
            # q_label shape, [nsamples, nparticle, DIM]

            if __debug__:  # any elements have nan then become true ,so that get error
                assert ((input[:, 2, :, :] == label[:, 2, :, :]).any() == True), 'error .. not match boxsize ...'

            valid_predict = (q_predict, p_predict)
            valid_label = (q_label, p_label)

            loss,qloss,ploss = self._loss(valid_predict, valid_label, self._phase_space, self.w)

            valid_loss  += loss.item()   # get the scalar output
            valid_qloss += qloss.item()  # get the scalar output
            valid_ploss += ploss.item()  # get the scalar output

            self.pred_valid_time += end_pred - start_pred

        return valid_loss / (step+1), valid_qloss / (step+1), valid_ploss / (step+1)

    # ===================================================
    def loss_distribution(self) -> object:
        ''' function to train one epoch'''

        self.any_HNN.eval()
        train_loss  = []
        train_qloss = []
        train_ploss = []

        for step, (input, label) in enumerate(self.data_loader.train_loader):

            # input shape, [nsamples, (q,p)=2, nparticle, DIM]
            self._phase_space.set_q(input[:, 0, :, :])
            self._phase_space.set_p(input[:, 1, :, :])
            self._phase_space.set_l_list(input[:, 2, :, :])

            qpl_list, crash_idx = self.linear_integrator.one_step(self.any_HNN, self._phase_space, self.tau_cur)
            # qp_list shape, [nsamples, (q,p)=2, nparticle, DIM]

            q_predict = qpl_list[:, 0, :, :]; p_predict = qpl_list[:, 1, :, :]
            # q_predict shape, [nsamples, nparticle, DIM]

            q_label   = label[:, 0, :, :]  ; p_label   = label[:, 1, :, :]
            # q_label shape, [nsamples, nparticle, DIM]

            if __debug__:  # any elements have nan then become true ,so that get error
                assert ((input[:, 2, :, :] == label[:, 2, :, :]).any() == True), 'error .. not match boxsize ...'

            train_predict = (q_predict, p_predict)
            train_label = (q_label, p_label)

            qploss,qloss,ploss = self._loss(train_predict, train_label, self.w)

            train_loss.append(qploss.item())   # get the scalar output
            train_qloss.append(qloss.item())  # get the scalar output
            train_ploss.append(ploss.item())  # get the scalar output

        return train_loss,  train_qloss, train_ploss

    # ===================================================

    def nepoch(self, nepoch, write_chk_pt_basename, write_loss_filename, checkpoint_interval):

        ''' function to train and valid more than one epoch

        parameters
        -------
        write_chk_pt_basename : filename to save checkpoints
        write_loss_filename   : path + filename to save loss values

        Returns
        -------
        float
            train loss, valid loss every epoch
        '''

        text = ''

        for e in range(1, nepoch+1 ):

            start_epoch = time.time()
            ##################################################################
            train_loss,train_qloss,train_ploss, train_regloss = self.train_one_epoch()
            # nn_train_dt = self.any_HNN.show_total_nn_time()
            if self._opt.param_groups[0]['lr'] <= self.lr_thrsh:
                print('reset sch lr ....', self.lr_thrsh)
                self._opt.param_groups[0]['lr'] = self.lr_thrsh
            else:
                self._sch.step() # learning
            # train_aveRx1, train_aveRy1 = self.any_HNN.get_RxRy_dhdq1()
            # train_aveRx2, train_aveRy2 = self.any_HNN.get_RxRy_dhdq2()
            ###################################################################

            with torch.no_grad(): # reduce memory consumption for computations
                valid_loss, valid_qloss, valid_ploss = self.valid_one_epoch()
                # nn_valid_dt = self.any_HNN.show_total_nn_time()
                # valid_aveRx1, valid_aveRy1 = self.any_HNN.get_RxRy_dhdq1()
                # valid_aveRx2, valid_aveRy2 = self.any_HNN.get_RxRy_dhdq2()
            ###################################################################
            end_epoch = time.time()
            #print('timing measure for one epoch')
            #print('train epoch: {:.6f}'.format(train_epoch_time),'valid epoch: {:.6f}'.format(valid_epoch_time),
            #      'train predicted: {:.6f}'.format(self.pred_train_time),'valid predicted: {:.6f}'.format(self.pred_valid_time),
            #      'train dhdq: {:.6f}'.format( nn_train_dt), 'valid dhdq: {:.6f}'.format(nn_valid_dt), 'backward:{:.6f}'.format(self.backward_time))

            if e%checkpoint_interval == 0:
                this_filename = write_chk_pt_basename + str(e) + '.pth'
                self.chk_pt.save_checkpoint(this_filename)

            dt = end_epoch - start_epoch

            text = text + str(e) + ' ' + str(train_loss)  + ' ' + str(valid_loss)  + \
                   ' ' + str(train_qloss)  + ' ' + str(valid_qloss) + \
                   ' ' + str(train_ploss) + ' '  + str(valid_ploss) + \
                   ' ' + str(train_regloss) + ' ' + str(self._opt.param_groups[0]['lr']) +\
                   ' ' + str(dt) + '\n'

            if e%MD_learner._verbose_interval==0:
                print('{} epoch:'.format(e), 'train_loss:{:.6f}'.format(train_loss),
                      'valid_loss:{:.6f}'.format(valid_loss),  'each epoch time:{:.5f}'.format(dt))
                print('optimizer lr {:.5f}'.format(self._opt.param_groups[0]['lr']),
                      ' train_dq {:.6f}'.format(train_qloss),' valid_dq {:.6f}'.format(valid_qloss),
                      ' train_dp {:.6f}'.format(train_ploss),' valid_dp {:.6f}'.format(valid_ploss),
                      'reg_loss {:5f}'.format(train_regloss), flush=True)

                mem = self.system_logs.record_memory_usage('e',e)
                self.save_loss_curve(text, write_loss_filename) 

    # ===================================================
    def save_loss_distribution(self, write_loss_filename):
        train_loss,  train_qloss, train_ploss = self.loss_distribution()
        torch.save(train_loss, write_loss_filename)
    # ===================================================
    def save_loss_curve(self, text, write_loss_filename):
        ''' function to save the loss every epoch '''

        with open(write_loss_filename, 'w') as fp:
            fp.write(text)
        fp.close()

