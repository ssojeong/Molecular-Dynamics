import copy
import torch.nn as nn
import torch

class SingleParticleTransformerNet(nn.Module):

    def __init__(self, input_dim, output_dim, traj_len, ngrids, d_model, nhead, n_encoder_layers, p):
        super().__init__()

        self.traj_len = traj_len
        self.ngrids = ngrids
        self.output_dim = output_dim
        self.prep_output_dim = round(input_dim / self.ngrids / self.traj_len)
        #20250803: print shape
        print('single particle transformer net : d_model',d_model,'output_dim',output_dim)
        print('single-particle tranformernet : ngrid', self.ngrids, 'input dim', input_dim, 'traj len', self.traj_len, 'prep output dim', self.prep_output_dim)
        self.feat_embedder = nn.Linear(self.ngrids * self.prep_output_dim, d_model)        # 4 for 2d of p & q
        self.pos_embed = nn.Parameter(torch.randn(1, traj_len, d_model) * .02)
        #assert input_dim == self.ngrids * 4 * self.traj_len
        self.next_pt = nn.Parameter(torch.randn(1, 1, d_model) * .02)

        print('!!!!! single par transformer_net', input_dim, output_dim, self.prep_output_dim, traj_len, ngrids)

        # use default setting: activation function=relu, dropout=0.1
        if n_encoder_layers > 0:
            self.transformer = nn.Sequential(*[EncoderLayer(d_model, nhead, p) for _ in range(n_encoder_layers)])
        else:
            self.transformer = nn.Identity()

    @staticmethod
    def weight_range():
        print('No weight range check for transformer')

    def forward(self, x):
        # input x.shape [nsample * nparticle, traj_len, ngrid * DIM * (q,p)]
        # q_prev shape [nsamples,nparticles,2]

        x = self.feat_embedder(x)               # shape: [nsample * nparticle, traj_len, d_model]
        x = x + self.pos_embed                  # add position info, same shape as above
        x = torch.cat([x, self.next_pt.expand(x.size(0), -1, -1)], dim=1)  # shape: [nsamples * nparticles, traj_len+1, d_model]
        x = self.transformer(x)                 # same shape as above
        x = x[:, -1, :]                         # shape: [nsample * nparticle, output_dim]
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
        # shape: [nsamples * nparticles, traj_len+1, d_model]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # 3 : qkv
        # shape [nsamples * nparticles, traj_len+1, (qkv), num_heads, head_dim]
        # permute -> shape [(qkv), nsamples * nparticles, num_heads, traj_len+1, head_dim]
        q, k, v = qkv.unbind(0)      # shape [nsamples * nparticles, num_heads, traj_len+1, head_dim]
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)      # shape [nsamples * nparticles, num_heads, traj_len+1, traj_len+1]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v                        # shape [nsamples * nparticles, num_heads, traj_len+1, head_dim]

        x = x.transpose(1, 2).reshape(B, N, C)  # shape [nsamples * nparticles, traj_len+1, num_heads * head_dim]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

