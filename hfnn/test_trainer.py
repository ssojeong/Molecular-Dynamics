import math
import unittest
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from MD.velocity_verlet import velocity_verlet
from ML.trainer.trainer import trainer
from utils.mydevice import mydevice


class TrainerTest(unittest.TestCase):
    train_config = {"loadfile": None,  # to load previously trained model
                    "nn_mode": 'ff',  # 'hf' or 'ff' predict hamiltonian or predict force
                    "force_clip": 5,  # clamp the force within a range
                    "grad_clip": 10,  # clamp the gradient for neural net parameters
                    "tau_long": 0.1,  # the large time step
                    "n_chain": 1,  # number of times to do integration before cal the loss
                    "ngrids": 6,  # for multibody interactions
                    "b": 0.2,  # grid lattice constant for multibody interactions
                    "lr": 1e-2,  # starting learning rate
                    "alpha_lr": 1e-4,  # lr for alpha for mbpw
                    "sch_step": 10,  # scheduler step
                    "sch_decay": 0.98}  # scheduler decay

    loss_config = {"eweight": 1,
                   "polynomial_degree": 4}

    input_dimension = 25
    output_dimension = 2
    num_samples = 2
    num_particles = 25
    num_dimensions = 2

    @classmethod
    def setUpClass(cls) -> None:
        mydevice()

        cls._trainer = trainer(cls.train_config, cls.loss_config)

    def test_forwarding(self) -> None:
        random_tensor: Tensor = torch.rand(1, self.input_dimension)
        result: Tensor = self._trainer.net_list[2].forward(random_tensor)
        (first_dimension_length, second_dimension_length) = result.size()
        self.assertEqual(second_dimension_length, self.output_dimension, "output dimension is not the same")

        for output in result[0]:
            self.assertNotEqual(output, float('nan'), 'Found nan in the result')

    def _random_tensor(self, dimension: Tuple) -> Tensor:
        while True:
            yield torch.rand(*dimension, requires_grad=True)

    def test_loss_metric(self) -> None:
        random_tensor_generator_3d = self._random_tensor((self.num_samples, self.num_particles, self.num_dimensions))
        loss_val = self._trainer.loss_obj.eval(q_list=next(random_tensor_generator_3d),
                                               p_list=next(random_tensor_generator_3d),
                                               q_label=next(random_tensor_generator_3d),
                                               p_label=next(random_tensor_generator_3d),
                                               q_init=next(random_tensor_generator_3d),
                                               p_init=next(random_tensor_generator_3d),
                                               l_list=next(random_tensor_generator_3d))

        self._check_invalid_value(loss_val)

    def _check_invalid_value(self, value: Tensor) -> None:
        invalid_losses = [float('nan'), math.inf, np.inf, -math.inf, -np.inf, np.nan]

        size = value.size()
        is_scalar = len(size) == 0
        if is_scalar:
            for invalid_item in invalid_losses:
                self.assertNotEqual(value, invalid_item, f'Item founds to be {invalid_item}')
        else:
            for item in value:
                self._check_invalid_value(item)

    def test_loss_mlvv_autograd(self) -> None:
        random_tensor_generator_3d = self._random_tensor((self.num_samples, self.num_particles, self.num_dimensions))

        q_init = next(random_tensor_generator_3d)
        p_init = next(random_tensor_generator_3d)
        l_init = next(random_tensor_generator_3d)
        q_label = next(random_tensor_generator_3d)
        p_label = next(random_tensor_generator_3d)

        mlvv: velocity_verlet = self._trainer.mlvv
        q_predict, p_predict, l_init = mlvv.nsteps(q_list=q_init, p_list=p_init, l_list=l_init,
                                                   tau=self.train_config['tau_long'],
                                                   n_chain=self.train_config['n_chain'])

        loss_val = self._trainer.loss_obj.eval(q_predict, p_predict, q_label, p_label, q_init, p_init, l_init)
        loss_val.backward()

        for item in [q_init.grad, loss_val, p_init.grad]:
            self._check_invalid_value(item)

    def test_weight(self) -> None:
        random_tensor_generator_3d = self._random_tensor((self.num_samples, self.num_particles, self.num_dimensions))

        q_init = next(random_tensor_generator_3d)
        p_init = next(random_tensor_generator_3d)
        l_init = next(random_tensor_generator_3d)
        q_label = next(random_tensor_generator_3d)
        p_label = next(random_tensor_generator_3d)

        self._trainer.one_step(q_init=q_init, p_init=p_init, q_label=q_label, p_label=p_label, l_init=l_init)
        for neural_network in self._trainer.net_list:
            self._check_parameter(neural_network)

    def _check_parameter(self, neural_network) -> None:
        lower_bound = -5
        upper_bound = 5

        for param in neural_network.parameters():
            self._check_invalid_value(param)

            out_of_bound_entries = sum(p > upper_bound or p < lower_bound for p in param.flatten())
            self.assertEqual(out_of_bound_entries, 0,
                             f'Some entries are outside the range of {lower_bound} to {upper_bound}')


if __name__ == '__main__':
    unittest.main()
