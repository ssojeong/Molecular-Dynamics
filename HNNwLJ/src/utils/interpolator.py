import torch

class interpolator: 

    _obj_count = 0

    def __init__(self):

        interpolator._obj_count += 1
        assert (interpolator._obj_count == 1),type(self).__name__ + " has more than one object"

    def inverse_distance_interpolator(self, predict, q_list):

        return


