import torch
import numpy as np

class momentum_sampler:

    def __init__(self, temp, nsamples, nparticle, mass):

        self.vel = np.zeros((nsamples, nparticle, DIM))

    def momentum_samples(self):
        # 'generate': 'maxwell'
        sigma = np.sqrt(temp)  # sqrt(kT/m)

            # vx = np.random.normal(0.0, sigma, nparticle)
            # vy = np.random.normal(0.0, sigma, nparticle)
            # vel_xy = np.stack((vx, vy), axis=-1)
        self.vel = np.random.normal(0, 1, (self.vel.shape[1],self.vel.shape[2]))*sigma # make sure shape correct
        momentum = torch.tensor(vel) * mass

        return momentum

