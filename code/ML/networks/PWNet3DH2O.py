import torch.nn as nn
import torch


class PWNet3DH2O(nn.Module):

    # input is torch.cat(dq_sq, dp_sq)

    def __init__(self, input_dim, output_dim, nnodes):
        super().__init__()

        self.group = input_dim
        self.group_conv1d_layers = []
        self.channel_per_group = nnodes[-1]

        # Hidden widths are expressed as multiples of `groups`
        if isinstance(nnodes, int):
            hidden_dims = [nnodes * self.group]
        else:
            hidden_dims = [h * self.group for h in nnodes]

        layers = []
        in_chs = [input_dim] + hidden_dims[:-1]
        out_chs = hidden_dims
        for cin, cout in zip(in_chs, out_chs):
            layers.append(nn.Conv1d(cin, cout, kernel_size=1, groups=self.group))

        self.group_convs = nn.ModuleList(layers)
        self.last_layer = nn.Conv1d(hidden_dims[-1], output_dim, kernel_size=1)

        # Numerical stabilizers/params as buffers, so they move with .to(device)
        self.register_buffer("epsilon", torch.tensor(1e-1, dtype=torch.float32))
        self.inv_max_expon = 3  # keep as Python int (used in exponent)

    def factor(self, r_square):
        """
        r_square: (nsamples * nparticles * nparticles, 1, ngrids) nonnegative
        """
        return 1.0 / (r_square**self.inv_max_expon + self.epsilon)

    def forward(self, x):
        """
         x:    (nsamples * nparticles * nparticles, input_dim, ngrids)
         mask: (nsamples * nparticles * nparticles, input_dim * hidden_dims[-1], ngrids)
         return: (nsamples * nparticles * nparticles, output_dim, ngrids)
         """
        mask = (x > 0.000001).float()
        assert torch.sum(mask) == x.size(0) * x.size(2),\
            f'ones in mask should be {x.size(0) * x.size(2)}, got {torch.sum(mask)}'
        mask = mask.repeat_interleave(self.channel_per_group, dim=1)
        r_square = x
        for m in self.group_convs:
            x = m(x)
            x = torch.relu(x)
        x = x * mask
        x = self.last_layer(x)
        x = torch.tanh(x)   # shape: (nsamples * nparticles * nparticles, output_dim, ngrids)
        # r_square must be [ 0, 0, r, 0, ...
        w = self.factor(torch.sum(r_square, dim=1)).unsqueeze(1)
        # w shape: (nsamples * nparticles * nparticles, 1, ngrids)
        # print(x.shape, w.shape)
        return x * w
