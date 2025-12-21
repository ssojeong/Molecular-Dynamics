# track changes
# LW 20250325

import torch
import torch.nn as nn

# import sys
# sys.path.append('../../')

from SingleParticleTransformerNet  import EncoderLayer

# Main test block
if __name__ == '__main__':

    d_model = 512
    dim_feedforward = 2048
    nhead = 8

    pytorch_encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=0,
        batch_first=True,
        norm_first=True,
        activation='gelu'
    ).to(torch.float64)

    print(pytorch_encoder_layer.state_dict().keys())
    # Homemade encoder
    homemade_encoder = EncoderLayer(d_model, nhead, p=0, qkv_bias=True).to(dtype=torch.float64)
    print(homemade_encoder.state_dict().keys())

    # Copy weights
    with torch.no_grad():
        torch_dict = pytorch_encoder_layer.state_dict()
        home_dict = homemade_encoder.state_dict()

        home_dict['attn.qkv.weight'] = torch_dict['self_attn.in_proj_weight']
        home_dict['attn.qkv.bias']   = torch_dict['self_attn.in_proj_bias']
        home_dict['attn.proj.weight'] = torch_dict['self_attn.out_proj.weight']
        home_dict['attn.proj.bias']   = torch_dict['self_attn.out_proj.bias']
        #
        home_dict['mlp.0.weight'] = torch_dict['linear1.weight']
        home_dict['mlp.0.bias']   = torch_dict['linear1.bias']
        home_dict['mlp.3.weight'] = torch_dict['linear2.weight']
        home_dict['mlp.3.bias']   = torch_dict['linear2.bias']
        #
        home_dict['norm1.weight'] = torch_dict['norm1.weight']
        home_dict['norm1.bias']   = torch_dict['norm1.bias']
        home_dict['norm2.weight'] = torch_dict['norm2.weight']
        home_dict['norm2.bias']   = torch_dict['norm2.bias']

        homemade_encoder.load_state_dict(home_dict)

    # Compare outputs
    x = torch.rand((128, 100, d_model), dtype=torch.float64)  # [B, T, D]
    y_torch = pytorch_encoder_layer(x)
    y_home = homemade_encoder(x)

    error = torch.sum(torch.abs(y_torch - y_home)).item()
    assert error < 1e-5, f"===== failed in test_transformer_encoder_wrapper.py Error: {error: .3e}"
    print(f"test_transformer_encoder_wrapper.py . . . passed")
