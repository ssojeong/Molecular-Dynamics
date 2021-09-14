from phase_space.pb import pb

class phase_space(pb):

    ''' phase space class that have q, p, ml_dhdq1, ml_dhdq2, and boxsize
        copy q_list to _q_list and p_list to _p_list
        copy ml_dhdq1 to _ml_dhdq1 and ml_dhdq2 to _ml_dhdq2
        the main reason to set ml_dhdq1 and ml_dhdq2 is for keeping them to get configuration
        before crash in crash_chkr
    '''

    _obj_count = 0

    def __init__(self):

        super().__init__()

        phase_space._obj_count += 1
        assert(phase_space._obj_count <= 3), type(self).__name__ + ' has more than two objects'
        # one phase space object for the whole code
        # the other phase space object only use as a copy in lennard-jones class in dimensionless
        # form

        '''initialize phase space of [nsamples, nparticle, DIM] '''
        self._q_list = None
        self._p_list = None
        self._ml_dhdq1_list = None
        self._ml_dhdq2_list = None
        self._boxsize = None
        print('phase_space initialized')

    def set_p(self, p_list):
        self._p_list = p_list.clone()

    def set_q(self, q_list):
        self._q_list = q_list.clone()

    def set_ml_dhdq1(self, ml_dhdq1_list):
        self._ml_dhdq1_list = ml_dhdq1_list.clone()

    def set_ml_dhdq2(self, ml_dhdq2_list):
        self._ml_dhdq2_list = ml_dhdq2_list.clone()

    def set_boxsize(self,boxsize):
        self._boxsize = boxsize

    def get_p(self):
        return self._p_list.clone()

    def get_q(self):
        return self._q_list.clone()

    def get_ml_dhdq1(self):
        return self._ml_dhdq1_list.clone()

    def get_ml_dhdq2(self):
        return self._ml_dhdq2_list.clone()

    def get_boxsize(self):
        return self._boxsize

