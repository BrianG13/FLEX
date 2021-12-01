import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from base.base_model import base_model

N_HEADS = 64
N_LAYERS = 2

"""
Some of the code taken from: https://github.com/lucidrains/vit-pytorch/blob/f196d1ec5b52edf554031c4a9c977d3a4e40ec9d/vit_pytorch/vit.py#L79
"""


class TransformerModel(base_model):
    def __init__(self, config, input_dim, output_dim, n_heads=N_HEADS, n_layers=N_LAYERS):
        super(TransformerModel, self).__init__()
        self.config = config
        if config.arch.transformer_mode == "encoder":
            print(f'Creating transformer encoder with {n_heads} Heads & {n_layers} Layers')
            encoder_layers = nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dim_feedforward=output_dim)
            encoder_norm = nn.LayerNorm(output_dim)
            self.transformer_model = nn.TransformerEncoder(encoder_layers, num_layers=n_layers, norm=encoder_norm)
        elif config.arch.transformer_mode == "mha":
            print(f'Creating MHA layer with {n_heads} Heads')
            self.transformer_model = nn.MultiheadAttention(embed_dim=config.arch.channel, num_heads=n_heads)
        else:
            raise Exception('NO VALID CONFIGURATION FOR TRANSFORMER MODEL!')
        self.dummy_token = torch.nn.Parameter(torch.zeros((1, 1, config.arch.channel)))
        self.register_parameter(param=self.dummy_token, name='dummy_token')
        xavier_uniform_(self.dummy_token)

    def forward(self, x):
        # x.shape = [B,C,F,V]
        self.transformer_model = self.transformer_model
        bsz = x.shape[0]
        channels = x.shape[1]
        frames = x.shape[2]
        views = x.shape[3]
        x = x.transpose(1, 3)  # x.shape = [B,V,F,C]
        x = x.transpose(1, 2)  # x.shape = [B,F,V,C]
        x = x.contiguous().view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # x.shape = [B*F,V,C]
        x = x.transpose(0, 1)  # x.shape = [V,B*F,C]

        temp = self.dummy_token.expand(-1, x.shape[1], -1)  # Expand token to B*F dimension
        x = torch.cat((temp, x), dim=0)  # x.shape = [B*F,V+1,C]

        if self.config.arch.transformer_mode == "mha":
            x, attn_output_weights = self.transformer_model(x, x, x)
        else:
            x = self.transformer_model(x)

        x = x.transpose(0, 1)  # [B*F, 1 ,C]
        x = x.contiguous().view(bsz, frames, views + 1, channels)  # [B,F,V+1,C]
        x = x.transpose(1, 2)  # [B,F,V,C]
        x = x.transpose(1, 3)  # [B,C,F,V]
        x = x[:, :, :, 0:1]  # [B,C,F,1]

        return x
