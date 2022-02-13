import torch
from inspect import currentframe, getframeinfo

# ===================================================
def assert_nan(x):
    cframe = currentframe().f_back
    filename = getframeinfo(cframe).filename
    lineno = cframe.f_lineno
    masknan = torch.isnan(x)
    if masknan.any() == True:
        print(filename,' line ',lineno,' has nan')
        quit()
 
# ===================================================




