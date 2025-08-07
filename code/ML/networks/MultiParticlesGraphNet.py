import torch
import torch.nn as nn
from utils.mydevice import mydevice


class MultiParticlesGraphNet(nn.Module):

    def __init__(self, input_dim, output_dim, n_gnn_layers, act_fn=nn.GELU(), attention=False, residual=False):
        print('!!!!! multi par graph net ', input_dim, output_dim, n_gnn_layers, 'attention', attention)
        super(MultiParticlesGraphNet, self).__init__()

        self.hidden_nf = input_dim
        self.output_dim = output_dim
        self.n_gnn_layers = n_gnn_layers

        if n_gnn_layers > 0:
            for i in range(0, self.n_gnn_layers):
                self.add_module("gcl_%d" % i, GNNLayer(self.hidden_nf, self.hidden_nf, self.hidden_nf,
                                                       act_fn=act_fn, residual=residual, attention=attention))
        else:
            self.add_module("gcl_0", nn.Identity())

        #self.readout = nn.Sequential(nn.Linear(self.hidden_nf, output_dim), nn.Tanh()) if readout else nn.Identity()


    def weight_range(self):
        print('No weight range check for transformer')

    def forward(self, h, coord): #20230701 -- SJ
        # h.shape [nsample * nparticle, hidden_nf]

        edges, coord = generate_fc_edges_batch(coord) # coord.shape [nsample * nparticle, 2]
        # return edges[0] shape nsample * npar * (npar -1)
        # return coord shape [nsample * npar, dim]

        coord_diff = coord[edges[0]] - coord[edges[1]]        # shape [nsample * nparticle * (nparticle - 1), dim]

        for i in range(0, self.n_gnn_layers):
            h = self._modules["gcl_%d" % i](h, edges, coord_diff)

        #h = self.readout(h)
        # shape  [nsample * nparticle, d_model]

        return h


class GNNLayer(nn.Module):

    def __init__(self, input_nf, output_nf, hidden_nf, act_fn, residual, attention):

        super(GNNLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(input_nf) # input_nf = hidden_nf
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
        # source, target, coord_diff => h[row], h[col], coord_diff
        #print('!!! edge model', source.shape, target.shape, coord_diff.shape)
        out = torch.cat([source, target, self.coord_mlp(coord_diff)], dim=1)

        out = self.edge_mlp(out)
        if self.attention:      # numerically checked
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, h, row, edge_feat):
        # h.shape [nsample * nparticle, hidden_nf]
        # row.shape nsample * nparticle * (nparticle -1)
        # edge_feat.shape [nsample * nparticle * (nparticle - 1), hidden_nf]

        agg = unsorted_segment_sum(edge_feat, row, num_segments=h.size(0))
        # agg.shape [nsample * nparticle, hidden_nf]
        agg = torch.cat([h, agg], dim=1)    # shape [nsample * nparticle, hidden_nf * 2]
        out = self.node_mlp(agg)            # shape [nsample * nparticle, hidden_nf]
        if self.residual:
            out = h + out
        return out

    def forward(self, h, edge_index, coord_diff):  # layer( = input h) norm --- LW suggest
        # h.shape [nsample * nparticle, hidden_nf]
        # coord_diff.shape [nsample * nparticle * (nparticle - 1), dim]

        h = self.layer_norm(h)
        row, col = edge_index
        # edge_index[0].shape nsample * nparticle * (nparticle -1)

        edge_feat = self.edge_model(h[row], h[col], coord_diff)
        # edge_feat.shape is [nsample * nparticle * (nparticle - 1), hidden_nf]

        h = self.node_model(h, row, edge_feat)

        return h


def unsorted_segment_sum(src, index, num_segments):      # numerically checked
    # src = edge_feat shape [nsample * nparticle * (nparticle - 1), hidden_nf] : the source tensor
    # index = row shape [nsample * nparticle * (nparticle -1)] : the indices of elements to scatter
    # num_segments = h.size(0) [nsample * nparticle]

    tgt_shape = (num_segments, src.size(1))
    # shape [nsample * nparticle, hidden_nf]
    tgt = src.new_full(tgt_shape, 0)         # Init empty result tensor.

    index = index.unsqueeze(-1).expand(-1, src.size(1))
    # index shape [nsample * nparticle * (nparticle -1), hidden_nf]
    index =  mydevice.load(index)

    tgt.scatter_add_(dim=0, index=index, src=src)
    # tgt.shape [nsample * nparticle, hidden_nf]

    return tgt


def generate_fc_edges(n_nodes):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)      # tgt
                cols.append(j)      # src
    edges = [rows, cols]
    # len(rows) = n_node * (n_nodes -1)
    # len(cols) = n_node * (n_nodes -1)
    return edges


def generate_fc_edges_batch(coord):         # !!! no self loop
    # coord shape [batch_size, npar, dim]
    batch_size, n_nodes = coord.size(0), coord.size(1)
    edges = generate_fc_edges(n_nodes)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    rows, cols = [], []
    for i in range(batch_size):
        rows.append(edges[0] + n_nodes * i)         # for uniform n_nodes only
        cols.append(edges[1] + n_nodes * i)
    edges = [torch.cat(rows), torch.cat(cols)]
    # edges[0] shape batch_size * npar * (npar -1)
    coord = coord.reshape(batch_size * n_nodes, coord.size(2))
    # coord shape [batch_size * npar, dim]
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
