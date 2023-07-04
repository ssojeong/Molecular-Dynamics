import torch
from HNN.HNN_base                       import HNN_base
import time

class pairwise_HNN(HNN_base):
    ''' pairwise_HNN class to learn dHdq and then combine with nodHdq '''

    _obj_count = 0

    def __init__(self, pairwise_MLP1, pairwise_MLP2, on_off_noML):
        super(pairwise_HNN,self).__init__(pairwise_MLP1, pairwise_MLP2)

        pairwise_HNN._obj_count += 1
        assert (pairwise_HNN._obj_count == 1),type(self).__name__ + " has more than one object"

        self.on_off_noML = on_off_noML
        self.tau_long = None # check give tau or not, check MD json file
        print('pairwise_HNN initialized ')

    # ===================================================
    def dHdq1(self, phase_space):
        ''' use this to update p in first step of velocity verlet
        set ml_dhdq1 in phase space and can use this to get configuration
        before crash in crash_chkr '''
        # print('dhdq1')
        noml_dhdq, ml_dhdq1 = self.dHdq_all(phase_space, self.net1)
        Rx, Ry = self.calculate_RxRy_ratio(noml_dhdq, ml_dhdq1)  
        self.sum_RxRy_dhdq1(Rx, Ry) 
        phase_space.set_ml_dhdq1(ml_dhdq1)  
        return noml_dhdq, ml_dhdq1
    # ===================================================
    def dHdq2(self, phase_space):   
        ''' use this to update p in third step of velocity verlet
        set ml_dhdq2 in phase space and can use this to get configuration
        before crash in crash_chkr '''
        # print('dhdq2')
        noml_dhdq,ml_dhdq2 = self.dHdq_all(phase_space, self.net2)
        Rx, Ry = self.calculate_RxRy_ratio(noml_dhdq, ml_dhdq2) 
        self.sum_RxRy_dhdq2(Rx, Ry)  
        phase_space.set_ml_dhdq2(ml_dhdq2)  
        return noml_dhdq, ml_dhdq2
    # ===================================================
    def dHdq_all(self, phase_space, net):
        ''' function to calculate dHdq = noML_dHdq + residual ML_dHdq

        Parameters
        ----------
        phase_space : contains q_list, p_list as input
                q_list shape is [nsamples, nparticle, DIM]
        net         : pass netA or netB

        Returns
        ----------
        corrected_dHdq : torch.tensor
                shape is [nsamples,nparticle,DIM]

        '''
        q_list = phase_space.get_q()

        nsamples, nparticle, DIM = q_list.shape

        if self.on_off_noML == 'on':
            noML_dHdq = self.dHdq(phase_space) # super().dHdq(phase_space)
            # noML_dHdq shape is [nsamples, nparticle, DIM]
        else:
            noML_dHdq = torch.zeros([nsamples, nparticle, DIM])  # not use noML in f-hn
            # noML_dHdq shape is [nsamples, nparticle, DIM]

        x = self.pack_dqdp_tau(phase_space, self.tau_long)
        # x.shape = [nsamples * nparticle * nparticle, 5]

        assert ( torch.isnan(x).any() == False ), 'input or label get nan......'
        #print('nn input', x)

        start = time.time()
        predict = net(x)
        # predict.shape = [nsamples * nparticle * nparticle, 2]
        end = time.time()
        self.dt += (end-start)

        predict2 = self.unpack_dqdp_tau(predict, q_list.shape)
        # predict2.shape = [nsamples, nparticle, DIM]

        #print('predict', predict2)

        # print('memory % used:', psutil.virtual_memory()[2], '\n')
        # print('corrected dHdq id ', id(corrected_dHdq))
        # print('noML dHdq id ', id(noML_dHdq))
        # print('predict id ', id(predict))

        return noML_dHdq, predict2

    # ===================================================
    def pack_dqdp_tau(self, phase_space, tau_cur):

        ''' function to prepare input in nn
        this function is use to make delta_q, delta_p, tau for input into models

        Parameters
        ----------
        tau_cur : float
                large time step for input in neural network
        phase_space : contains q_list, p_list as input for integration

        Returns
        ----------
        input in neural network
        here, 2*DIM + 1 is (del_qx, del_qy, del_px, del_py, tau )

        '''

        nsamples, nparticle, DIM = phase_space.get_q().shape

        dqdp_list = self.make_dqdp(phase_space)
        # dqdp_list.shape = [nsamples, nparticle, nparticle, 2*DIM]

        tau_tensor = torch.zeros([nsamples, nparticle, nparticle, 1], dtype=torch.float64)

        tau_tensor.fill_(tau_cur * 0.5)  # tau_tensor take them tau/2

        x = torch.cat((dqdp_list, tau_tensor),dim = 3)
        # x.shape = [nsamples, nparticle, nparticle, 2*DIM + 1]

        x = torch.reshape(x,(nsamples*nparticle*nparticle,5))
        # x.shape = [ nsamples * nparticle * nparticle, 2*DIM + 1]

        return x

    # ===================================================
    def unpack_dqdp_tau(self, y, qlist_shape):
        ''' function to make output unpack

        parameter
        _____________
        y  : predict
                y.shape = [ nsamples * nparticle * nparticle, 2]

        return
        _____________
        y2  : shape is  [nsamples,nparticle,2]
        '''

        nsamples, nparticle, DIM = qlist_shape

        y1 = torch.reshape(y, (nsamples, nparticle, nparticle, DIM))

        # check - run "python3.7 -O" to switch off
        #if __debug__:
        #    y2 = torch.clone(y)
        #    for i in range(nparticle): y2[:,i,i,:] = 0

        dy = torch.diagonal(y1,0,1,2) # offset, nparticle, nparticle
        torch.fill_(dy,0.0)

        #if __debug__:
        #    err = (y-y2)**2
        #    assert (torch.sum(err)<1e-6),'error in diagonal computations'

        y2 = torch.sum(y1, dim=2) # sum over dim =2 that is all neighbors
        # y.shape = [nsamples,nparticle,2]

        return y2
    # ===================================================
    def make_dqdp(self, phase_space):

        ''' function to make dq and dp for feeding into nn

        Returns
        ----------
        take q_list and p_list, generate the difference matrix
        q_list.shape = p_list.shape = [nsamples, nparticle, DIM]
        dqdp : here 4 is (dq_x, dq_y, dp_x, dp_y )
        '''

        q_list = phase_space.get_q()
        p_list = phase_space.get_p()

        #print('make dqdp', q_list, p_list)

        dq = self.delta_state(q_list)
        dp = self.delta_state(p_list)
        # dq.shape = dp.shape = [nsamples, nparticle, nparticle, 2]

        dqdp = torch.cat((dq, dp), dim=-1)
        # dqdp.shape is [ nsamples, nparticle, nparticle, 4]

        return dqdp
    # ===================================================

