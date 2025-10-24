import numpy as np


def translation(target_center, pos):
    mass_center = np.sum(pos * mass[None], axis=-1)
    print('current mass center')
    pos -= mass_center

if __name__ == "__main__":
    mass = np.asarray([16, 1, 1] * 8)
    print(mass)
