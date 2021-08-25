import torch
from datetime import datetime


class check4particle_crash_dummy:

    def __init__(self, rthrsh0, pthrsh0, rthrsh, pthrsh, crash_path):
        '''
        this class do nothing, use for case when we want to switch off check
        '''
        print('check4particle_crash_dummy initialized')
        return


    def check(self,phase_space, prev_q, prev_p, prev_dHdq1, prev_dHdq2, prev_pred1, prev_pred2, hamiltonian):
        # do no check and return
        return


