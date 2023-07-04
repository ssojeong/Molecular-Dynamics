import torch.nn as nn
import torch
from ML.networks.base_net import base_net


class mb_transformer4gnn_net(nn.Module):

    def __init__(self, input_dim, output_dim, traj_len, ngrids, d_model, nhead, n_encoder_layers, p, readout=True):
        super().__init__()

        print('mb transformer for gnn input readout ', readout )
        self.traj_len = traj_len
        self.ngrids = ngrids
        self.output_dim = output_dim
        self.pos_embed = nn.Parameter(torch.randn(1, traj_len, d_model) * .02)
        assert input_dim == self.ngrids * 4 * self.traj_len
        self.feat_embedder = nn.Linear(self.ngrids * 4, d_model)        # 4 for 2d of p & q
        self.next_pt = nn.Parameter(torch.randn(1, 1, d_model) * .02)

        # use default setting: activation function=relu, dropout=0.1
        if n_layers > 0:
            self.transformer = nn.Sequential(*[EncoderLayer(d_model, nhead, p) for _ in range(n_encoder_layers)])
        else:
            self.transformer = nn.Identity()
        self.readout = nn.Sequential(nn.Linear(d_model, output_dim), nn.Tanh()) if readout else nn.Identity()

    @staticmethod
    def weight_range():
        print('No weight range check for transformer')

    def forward(self, x):
        # x.shape [nsample * nparticle, ngrid * (q,p) * DIM * traj_len]
        x = x.reshape(x.size(0), self.ngrids * 4, self.traj_len).permute(0, 2, 1)
        # x.shape [nsample * nparticle, traj_len, ngrid * (q,p) * DIM]
        x = self.feat_embedder(x)               # shape: [nsample * nparticle, traj_len, d_model]
        x = x + self.pos_embed                  # add position info, same shape as above
        x = torch.cat([x, self.next_pt.expand(x.size(0), -1, -1)], dim=1)
        x = self.transformer(x)                 # same shape as above
        x = self.readout(x[:, -1, :])           # shape: [nsample * nparticle, output_dim/d_model]
        return x


class EncoderLayer(nn.Module):

    def __init__(self, dim, nhead, p, mlp_ratio=4, qkv_bias=False, qk_norm=False,
                 act_fn=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiheadAttention(dim=dim, nhead=nhead, p=p, qkv_bias=qkv_bias,
                                       qk_norm=qk_norm, norm_layer=norm_layer)

        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio),
                                 act_fn(),
                                 nn.Dropout(p),
                                 nn.Linear(dim * mlp_ratio, dim),
                                 act_fn(),
                                 nn.Dropout(p))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MultiheadAttention(nn.Module):

    def __init__(self, dim, nhead, p, qkv_bias, qk_norm, norm_layer):
        super().__init__()
        assert dim % nhead == 0, 'dim should be divisible by num_heads'
        self.num_heads = nhead
        self.head_dim = dim // nhead
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(p)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)     # shape: B, num_heads, N, head_dim
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)      # shape: B, num_heads, N, N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v                        # shape: B, num_head, N, head_dim

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GNNLayer(nn.Module):

    def __init__(self, input_nf, output_nf, hidden_nf, act_fn, residual, attention):
        super(GNNLayer, self).__init__()
        self.residual = residual
        self.attention = attention

        self.coord_mlp = nn.Sequential(
            nn.Linear(2, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_nf * 3, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))        # only one round act_fn, following EGNN

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, coord_diff):
        # print('!!!', source.shape, target.shape, coord_diff.shape)
        out = torch.cat([source, target, self.coord_mlp(coord_diff)], dim=1)
        # print(out.shape, source.shape)
        out = self.edge_mlp(out)
        if self.attention:      # numerically checked
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, h, row, edge_feat):
        # h.shape [nsample * nparticle, hidden_nf], edge_feat.shape [nsample * nparticle * (nparticle - 1), hidden_nf]
        agg = unsorted_segment_sum(edge_feat, row, num_segments=h.size(0))
        agg = torch.cat([h, agg], dim=1)    # shape [nsample * nparticle, hidden_nf * 2]
        out = self.node_mlp(agg)            # shape [nsample * nparticle, hidden_nf]
        if self.residual:
            out = h + out
        return out

    def forward(self, h, edge_index, coord_diff):
        row, col = edge_index
        # coord_diff = coord[row] - coord[col]        # shape [nsample * nparticle * (nparticle - 1), 2]

        edge_feat = self.edge_model(h[row], h[col], coord_diff)     # [nsample * nparticle * (nparticle - 1), hidden_nf]
        h = self.node_model(h, row, edge_feat)

        return h


class mb_gnn_net(nn.Module):

    def __init__(self, output_dim, n_gnn_layers, d_model,  act_fn=nn.GELU(), residual=False, attention=False):

        super(mb_gnn_net, self).__init__()
        self.hidden_nf = d_model
        self.n_gnn_layers = n_gnn_layers
        self.readout = nn.Sequential(nn.Linear(self.hidden_nf, output_dim), nn.Tanh())

        # 20230701 -- SJ
        #self.transformer = mb_transformer_net(input_dim, self.hidden_nf, traj_len, ngrids, d_model, nhead,
        #                                      n_trans_layers, p, readout=False)

        for i in range(0, self.n_gnn_layers):
            self.add_module("gcl_%d" % i, GNNLayer(self.hidden_nf, self.hidden_nf, self.hidden_nf,
                                                   act_fn=act_fn, residual=residual, attention=attention))

    @staticmethod
    def weight_range(self):
        print('No weight range check for transformer')

    def forward(self, input_net, h, coord): #20230701 -- SJ
        # h.shape [nsample * nparticle, ngrids * DIM * (q,p) * traj_len], coord.shape [nsample, nparticle, 2]
        h = input_net(h)
        # h.shape [nsample * nparticle, hidden_nf]
        edges, coord = generate_fc_edges_batch(coord)
        # edges [row_list, col_list], coord.shape [nsample * nparticle, 2]
        coord_diff = coord[edges[0]] - coord[edges[1]]        # shape [nsample * nparticle * (nparticle - 1), 2]
        for i in range(0, self.n_gnn_layers):
            h = self._modules["gcl_%d" % i](h, edges, coord_diff)
        h = self.readout(h)
        return h


def unsorted_segment_sum(src, index, num_segments):      # numerically checked
    tgt_shape = (num_segments, src.size(1))
    tgt = src.new_full(tgt_shape, 0)         # Init empty result tensor.
    index = index.unsqueeze(-1).expand(-1, src.size(1))
    tgt.scatter_add_(dim=0, index=index, src=src)
    return tgt


def generate_fc_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)      # tgt
                cols.append(j)      # src
    edges = [rows, cols]
    return edges


def generate_fc_edges_batch(coord):         # !!! no self loop
    batch_size, n_nodes = coord.size(0), coord.size(1)
    edges = generate_fc_edges(n_nodes)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    rows, cols = [], []
    for i in range(batch_size):
        rows.append(edges[0] + n_nodes * i)         # for uniform n_nodes only
        cols.append(edges[1] + n_nodes * i)
    edges = [torch.cat(rows), torch.cat(cols)]
    coord = coord.reshape(batch_size * n_nodes, coord.size(2))
    return edges, coord


# if __name__ == "__main__":
#     # Dummy parameters
#     batch_size_ = 2
#     n_nodes_ = 3
#     output_dim = 2
#     traj_len = 1
#     ngrids = 2
#     input_dim = ngrids * traj_len * 4
#     d_model = 16
#     nhead = 4
#     n_encoder_layers = 1
#     p = 0
#
#     # Dummy variables h, x and fully connected edges
#     h_ = torch.ones(batch_size_ * n_nodes_, input_dim)
#     coord_ = torch.arange(0, batch_size_ * n_nodes_ * 2).reshape(batch_size_, n_nodes_, 2).float()
#     # print(h_.shape, coord_.shape)
#
#     # Initialize GNN
#     gnn = mb_gnn_net(input_dim, output_dim, traj_len, ngrids, d_model, nhead, n_encoder_layers, n_gnn_layers=1,
#                      p=p, act_fn=nn.GELU(), residual=True, attention=True)
#
#     # Run GNN
#     h_ = gnn(h_, coord_)
#
#     print(h_)
