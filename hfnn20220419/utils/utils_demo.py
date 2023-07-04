from utils import assert_nan
import torch
import numpy

def g(x):
    assert_nan(x)

if __name__=='__main__':

   x = torch.rand([2])
   x[0] = numpy.sqrt(-1)
   g(x)
