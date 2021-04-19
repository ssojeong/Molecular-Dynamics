import torch.nn as nn
import torch

class fields_cnn(nn.Module):

    def __init__(self, gridL): # HK, 
        super(fields_cnn, self).__init__()

        self.gridL = gridL 

        self.correction_term = nn.Sequential(
            nn.Linear(self.gridL*self.gridL,self.gridL*self.gridL*2*2) # for test
        )
        self.correction_term.double()
        self.correction_term.apply(self.init_weights)
        print('fields_cnn initialized :  ', self.correction_term)
    # ============================================
    def init_weights(self,layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_normal_(layer.weight,gain=0.1) # SJ make weights small
            layer.bias.data.fill_(0.0)
    # ============================================
    def get_gridL(self):
        return self.gridL
    # ============================================
    def forward(self, x):

        nsamples, nchannels, gridx, gridy = x.shape
        # x.shape = [nsamples, gridL, gridL, nchannels=2]

        # x = F.flatten(x) # for test
        x = x.view(-1)
        MLdHdq = self.correction_term(x)

        MLdHdq = torch.reshape(MLdHdq,(nsamples,nchannels,gridx,gridy))

        return MLdHdq
