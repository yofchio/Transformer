import math

import torch
from torch import nn

from layer_norm import LayerNorm
from mlp import MLP
from multi_head_attention import MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norml = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.ffn = MLP(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, src_mask):
        h = x
        x = self.attention(x, x, x, src_mask)
        x = h + x
        x = self.norml(x)
        x = self.dropout1(x)

        h = x
        x = self.ffn(x)
        x = x + h
        x = self.norm2(x)
        x = self.dropout2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_layers, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, ffn_hidden, n_head, drop_prob)
            for _ in range(n_layers)
        ])

    def forward(self, src, src_mask):
        x = src
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


def main():
    encoder = Encoder(2, 128, 256, 4, 0.1)
    sample = torch.randn((5, 100, 128))
    print(encoder(sample, None).size())


if __name__ == '__main__':
    main()
