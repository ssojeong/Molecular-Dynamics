class A:
    def __init__(self):
        print('e6 in init ',e6)
        #e6 = 9 # gives error -- why cannot set it?

    def f(self):
        return e6


if __name__=='__main__':

    e6  = 5.0 # is this global?
    print('e6 in main ',e6)
    a = A()
    v = a.f()
    print('v ',v)

