from torch import nn

from layer_norm import LayerNorm
from mlp import MLP
from multi_head_attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.trg_self_attention = MultiHeadAttention(d_model, n_head)
        self.norml = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.src_trg_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)
        self.ffn = MLP(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, trg, enc_src, trg_mask, src_trg_mask):
        h = trg
        x = self.trg_self_attention(trg, trg, trg, mask=trg_mask)
        x = x + h
        x = self.norml(x)
        x = self.dropout1(x)

        h = x
        x = self.src_trg_attention(q=x, k=enc_src, v=enc_src, mask=src_trg_mask)
        x = x + h
        x = self.norm2(x)
        x = self.dropout2(x)

        h = x
        x = self.ffn(x)
        x = x + h
        x = self.norm3(x)
        x = self.dropout3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, n_layers, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, ffn_hidden, n_head, drop_prob)
            for _ in range(n_layers)
        ])

    def forward(self, trg, enc_src, trg_mask, src_trg_mask):
        x = trg
        for layer in self.layers:
            x = layer(x, enc_src, trg_mask, src_trg_mask)
        return x
