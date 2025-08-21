import time
import torch
import torch.nn as nn

from LLUF.AtomPairIndexer import AtomPairIndexer
from networks.PWNet3DH2O import PWNet3DH2O as PWNet_New


class PWNet_Old(nn.Module):

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
        mask = (x > 0).float()
        assert torch.sum(mask) == x.size(0) * x.size(2),\
            f'ones in mask should be {x.size(0) * x.size(2)}, got {torch.sum(mask)}'
        mask = mask.repeat_interleave(self.channel_per_group, dim=1)
        w = self.factor(torch.sum(torch.relu(x), dim=1)).unsqueeze(1)
        # w shape: (nsamples * nparticles * nparticles, 1, ngrids)
        for m in self.group_convs:
            x = m(x)
            x = torch.relu(x)
        x = x * mask
        x = self.last_layer(x)
        x = torch.tanh(x)   # shape: (nsamples * nparticles * nparticles, output_dim, ngrids)
        # print(x.shape, w.shape)
        return x * w


class OneBatch:
    def __init__(self, net):
        super().__init__()
        self.net = net

    def __call__(self, x, flag):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        y = self.net(x)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        loss = torch.sum(y)
        loss.backward()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        if flag:
            return t1, t2, t3
        else:
            pass


if __name__ == "__main__":
    from einops import rearrange, repeat

    indexer = AtomPairIndexer(n_mol=8)
    pwnet_old = PWNet_Old(8, 3, [128, 128, 128])
    pwnet_new = PWNet_New(8, 3, [128, 128, 128])
    one_batch_old = OneBatch(pwnet_old)
    one_batch_new = OneBatch(pwnet_new)

    x = torch.rand(64, 24, 24, 6)**2
    print(x.requires_grad)
    # warm up
    for i in range(5):
        x_embed = indexer.fill_tensor(x)
        x_embed = rearrange(x_embed, 'ns i j c g -> (ns i j) c g', c=8)
        # print(x.shape, x_embed.shape)

        one_batch_old(x_embed, flag=False)
        one_batch_new(x_embed, flag=False)

    embedder_time_list = []
    forward_time_list = []
    backward_time_list = []
    for i in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # print(torch.sum(x > 0), x.shape)
        x_embed = indexer.fill_tensor(x)
        x_embed = rearrange(x_embed, 'ns i j c g -> (ns i j) c g', c=8)
        # print(torch.sum(x > 0), x.shape)
        t1, t2, t3 = one_batch_old(x_embed, flag=True)
        et = t1 - t0
        ft = t2 - t1
        bt = t3 - t2
        embedder_time_list.append(et)
        forward_time_list.append(ft)
        backward_time_list.append(bt)
    et_mean = torch.tensor(embedder_time_list).mean().item()
    ft_mean = torch.tensor(forward_time_list).mean().item()
    bt_mean = torch.tensor(backward_time_list).mean().item()

    print(f"Old PWNet3DH2O: Indexer {et_mean:.4f}, Forward {ft_mean:.4f}, Backward {bt_mean:.4f}")

    embedder_time_list = []
    forward_time_list = []
    backward_time_list = []
    for i in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x_embed = indexer.fill_tensor(x)
        x_embed = rearrange(x_embed, 'ns i j c g -> (ns i j) c g', c=8)
        t1, t2, t3 = one_batch_new(x_embed, flag=True)
        et = t1 - t0
        ft = t2 - t1
        bt = t3 - t2
        embedder_time_list.append(et)
        forward_time_list.append(ft)
        backward_time_list.append(bt)
    et_mean = torch.tensor(embedder_time_list).mean().item()
    ft_mean = torch.tensor(forward_time_list).mean().item()
    bt_mean = torch.tensor(backward_time_list).mean().item()

    print(f"New PWNet3DH2O: Indexer {et_mean:.4f}, Forward {ft_mean:.4f}, Backward {bt_mean:.4f}")
