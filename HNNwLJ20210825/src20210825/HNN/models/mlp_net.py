import torch.nn as nn

class mlp_net(nn.Module):

    def __init__(self,layer_list, dropout_list):
        super().__init__()

        nlayers = len(layer_list)
        self.layers = []
        for idx in range(nlayers-1):
            cur = layer_list[idx]
            nxt = layer_list[idx+1]
            h = nn.Linear(cur,nxt)
            self.layers.append(h)
            if idx<nlayers-2:
                #self.layers.append(nn.LeakyReLU(negative_slope=0.4))
                self.layers.append(nn.Tanh())
            if idx>0 and idx<nlayers-2:
                self.layers.append(nn.Dropout(p=dropout_list[idx]))

        self.layers  = nn.ModuleList(self.layers)
        self.layers.apply(self.init_weights) 

        print('MLP_net initialized : ',layer_list[0],'-> ... ->',layer_list[-2], '-> 2')
    # ============================================
    def init_weights(self,m): # m is layer that is nn.Linear
        if type(m) == nn.Linear:
            # set the xavier_gain neither too much bigger than 1, nor too much less than 1
            # recommended gain value for the given nonlinearity function
            #nn.init.xavier_normal_(m.weight,gain=nn.init.calculate_gain('leaky_relu',0.4)) # tanh gain=5/3
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))  # tanh gain=5/3
            m.bias.data.fill_(0.0)
    # ============================================
    def param_threshold(self,m): # HK
        # give parameter threshold to prevent exploding gradients
        # min-max from parameters that load trained model using original data as no crash data (no. 3,000,000)
        # -1.3077 < w < 1.2068  ,  -0.2449 < b < 0.1717
        if type(m) == nn.Linear:
            w = m.weight.data
            b = m.bias.data
            w = w.clamp(min=-3., max=3.)
            b = b.clamp(min=-1., max=1.)
            m.weight.data = w
            m.bias.data = b
    # ============================================
    def forward(self,x):
        # field_HNN -> x.shape is [nsamples*nparticle, grids18 + grids18 + 1]
        # pairwise_HNN -> x.shape is [nsamples * nparticle * nparticle, 5]]

        # open when give parameters thresholds
        #self.layers.apply(self.param_threshold)

        for ll in self.layers:
            x = ll(x)
            #print('layer output', ll, x)
            #if type(ll) == nn.Linear:
            #    print(ll, 'weights', ll.weight)
            #    print(ll, 'bias', ll.bias)

        # field_HNN -> x.shape is [nsamples*nparticle,2]
        # pairwise_HNN -> x.shape is [nsamples * nparticle * nparticle, 2]
        return x
    # ============================================
    def print_grad(self):
        for ll in self.layers:
            if type(ll) == nn.Linear:
                wv= ll.weight
                wg= ll.weight.grad
                bv= ll.bias
                bg= ll.bias.grad
                print('w ', wv,' grad ',wg)
                print('b ', bv,' grad ',bg)