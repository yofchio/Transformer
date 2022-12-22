import numpy as np
import torch
from torch import nn


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()
        self.encoding = torch.zeros((max_len, d_model), device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, padding_idx, drop_prob, device):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = PositionEmbedding(d_model, max_len, device)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        token_out = self.token_emb(x)
        pos_out = self.pos_emb(x)
        output = self.dropout(token_out + pos_out)
        return output


if __name__ == '__main__':
    emb = TransformerEmbedding(20, 128, 10, 1, 0.1, torch.device("cpu"))
    sample = torch.from_numpy(np.array([[2, 12, 3, 4, 5, 1, 1, 1],
                                        [2, 12, 3, 4, 5, 1, 1, 1],
                                        [2, 12, 3, 4, 5, 1, 1, 1],
                                        [2, 12, 3, 4, 5, 1, 1, 1],
                                        [2, 12, 3, 4, 5, 1, 1, 1]]))
    print(sample.shape)
    print(emb(sample).shape)
