import torch
import numpy as np
from ..Integrator.linear_integrator import linear_integrator
from .dataset import Hamiltonian_Dataset
from torch.utils.data import DataLoader
from ..hamiltonian.pb import periodic_bc
from ..phase_space import phase_space

class MD_learner:

    def __init__(self, **state):

        #self.linear_integrator = linear_integrator
        self.nepoch = state['epoch']
        self._optimizer = state['optim']
        self._loss = state['loss']

        try:  # data loader and seed setting
            self._batch_size = state['batch_size']  # will be used for data loader setting
            seed = state.get('seed', 937162211)  # default seed is 9 digit prime number

            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True  # Processing speed may be lower then when the models functions nondeterministically.
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            shuffle = state.get('shuffle', False)  # default shuffle the data loader
            num_workers = state.get('num_wokers', 0)

            self._sample = state['N']
            Temperature = state['Temperature']
            state['pb_q'] = periodic_bc()
            state['phase_space'] = phase_space()

        except:
            raise Exception('epoch / batch_size not defined ')

        DataLoader_Setting = {'num_workers': num_workers, 'pin_memory': True, 'shuffle': shuffle}

        self._train_dataset = Hamiltonian_Dataset(Temperature,
                                                  self._sample,
                                                  mode='train',
                                                  **state)

        self._train_loader = DataLoader(self._train_dataset,
                                        batch_size=self._batch_size,
                                        **DataLoader_Setting)

        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._current_epoch = 1

        state['_device'] = self._device
        state['tau'] = state['tau'] * state['iterations']  # large time step
        state['iterations'] = int(state['iterations'] / state['iterations'])  # one step
        # init_q, init_p = linear_integrator(**state).set_phase_space(nsamples = self._sample)
        #

        self._setting = state  # save the setting

        try:  # dataset setting
            if state['DIM'] != 2:
                raise Exception('Not supported for Dimension is not 2')
        except:
            raise Exception('Temperature_List for loading / sample not found ')

        # try:  # architecture setting
        #     self._model = kwargs['models'].double().to(self._device)
        # except:
        #     raise Exception('models not found')

    # phase_space consist of minibatch data
    # pb is boundary condition
    def train(self):

        print('set',self._setting)
        pairwise_hnn = self._setting['general_hamiltonian']
        pairwise_hnn.train()
        criterion = self._loss

        for e in range(self.nepoch):

            for batch_idx, data in enumerate(self._train_loader):
                print('batch_idx : {}, batch size : {}'.format(batch_idx, self._batch_size))

                print('=== initial data ===') # shape : nsamples x N_particles x DIM
                # q_list = data[0][0].to(self._device).requires_grad_(True)
                # p_list = data[0][1].to(self._device).requires_grad_(True)
                q_list = data[0][0].to(self._device)
                p_list = data[0][1].to(self._device)
                print(q_list,p_list)

                # to integrate
                # self._setting['pos'] = q_list.detach().cpu().numpy()  # convert tensor to numpy
                # self._setting['vel'] = p_list.detach().cpu().numpy()  # convert tensor to numpy
                self._setting['pos'] = q_list
                self._setting['vel'] = p_list

                print('=== label data ===')
                q_list_label = data[1][0].to(self._device)
                p_list_label = data[1][1].to(self._device)
                print(q_list_label,p_list_label)
                print('==================')

                label = (q_list_label, p_list_label)

                print('linear integrator')
                q_list_predict, p_list_predict = linear_integrator(**self._setting).integrate(pairwise_hnn, multicpu=False)
                q_list_predict = q_list_predict.reshape(-1,q_list_predict.shape[2],q_list_predict.shape[3])
                p_list_predict = p_list_predict.reshape(-1, p_list_predict.shape[2], p_list_predict.shape[3])

                # q_list_predict = torch.from_numpy(q_list_predict)
                # p_list_predict = torch.from_numpy(p_list_predict)

                pred = (q_list_predict, p_list_predict)
                pred = torch.tensor(pred, requires_grad=True)

                loss = criterion(pred, label)

                self._optimizer.zero_grad()  # defore the backward pass, use the optimizer object to zero all of the gradients for the variables
                loss.backward()  # backward pass : compute gradient of the loss wrt models parameters
                train_loss = loss.item()  # get the scalar output
                self._optimizer.step()

                print(e,train_loss)

    def step(self,phase_space,pb,tau):
        pairwise_hnn.eval()
        q_list_predict, p_list_predict = self.linear_integrator.integrate(**state)
        return q_list_predict,p_list_predict

    # def loss(self,...):