import copy
import torch.nn as nn
import torch
from ML.networks.base_net import base_net


class mb_transformer_net(base_net):

    def __init__(self, input_dim, output_dim, traj_len, ngrids, d_model, nhead, n_encoder_layers, p):
        super().__init__()

        self.traj_len = traj_len
        self.ngrids = ngrids
        self.output_dim = output_dim
        self.pos_embed = nn.Parameter(torch.randn(1, traj_len, d_model) * .02)
        self.feat_embedder = nn.Linear(input_dim // self.traj_len, d_model)
        self.next_pt = nn.Parameter(torch.randn(1, 1, d_model) * .02)

        # use default setting: activation function=relu, dropout=0.1
        self.transformer = nn.Sequential(*[EncoderLayer(d_model, nhead, p) for _ in range(n_encoder_layers)])
        self.readout = nn.Linear(d_model, output_dim)

    def weight_range(self):
        print('No weight range check for transformer .....')

    def forward(self, x, q_pre): # SJ coord
        # input x shape [nsample * nparticales, ngrids * DIM * (q,p) * traj_len]
        # q_prev shape [nsamples,nparticles,2]
        npar = q_pre.shape[1] # nparticles

        x = x.reshape(x.size(0), self.traj_len, self.ngrids * 4)
        # shape of x: [nsamples * nparticles, traj_len, ngrids * DIM * (q,p)]

        x = self.feat_embedder(x)               # shape: [nsamples * nparticles, traj_len, d_model]
        x = x + self.pos_embed                  # add position info, same shape as above
        x = torch.cat([x, self.next_pt.expand(x.size(0), -1, -1)], dim=1) # shape: [nsamples * nparticles, traj_len+1, d_model]
        x = self.transformer(x)                 # same shape as above
        x = self.readout(x[:, -1, :])           # shape: [nsamples * nparticles, output_dim]
        x = torch.tanh(x)                       # same shape as above
        return x

class EncoderLayer(nn.Module):

    def __init__(self, dim, nhead, p, mlp_ratio=4, qkv_bias=False, qk_norm=False,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiheadAttention(dim=dim, nhead=nhead, p=p, qkv_bias=qkv_bias,
                                       qk_norm=qk_norm, norm_layer=norm_layer)

        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * mlp_ratio),
                                 act_layer(),
                                 nn.Dropout(p),
                                 nn.Linear(dim * mlp_ratio, dim),
                                 act_layer(),
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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)    # 3 : qkv
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(p)

    def forward(self, x):
        B, N, C = x.shape
        # shape: [nsamples * nparticles, traj_len+1, d_model]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # 3 : qkv
        # shape [nsamples * nparticles, traj_len+1, (qkv), num_heads, head_dim]
        # permute -> shape [(qkv), nsamples * nparticles, num_heads, traj_len+1, head_dim]
        q, k, v = qkv.unbind(0)   # shape [nsamples * nparticles, num_heads, traj_len+1, head_dim]
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)   # shape [nsamples * nparticles, num_heads, traj_len+1, traj_len+1]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v     # shape [nsamples * nparticles, num_heads, traj_len+1, head_dim]

        x = x.transpose(1, 2).reshape(B, N, C) # shape [nsamples * nparticles, traj_len+1, num_heads * head_dim]
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
