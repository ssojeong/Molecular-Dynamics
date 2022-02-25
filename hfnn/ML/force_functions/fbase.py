
class fbase:

    def __init__(self,net1,net2):
        assert net1 is not net2, f'net1 and net2 refers to the same object'
        self.net1 = net1
        self.net2 = net2

    def save(self,basename):

        pass



