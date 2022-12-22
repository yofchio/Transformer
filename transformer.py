import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from decoder import Decoder
from embedding import TransformerEmbedding
from encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, src_pad_idx,
                 trg_pad_idx, trg_sos_idx, src_voc_size, trg_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device=torch.device("cpu")):
        # 1, 1, 2, 10, 10, 128, 4, 10, 128, 2, 0.1
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx

        self.src_emb = TransformerEmbedding(src_voc_size, d_model, max_len, src_pad_idx, drop_prob, device)
        self.encoder = Encoder(n_layers, d_model, ffn_hidden, n_head, drop_prob)
        self.trg_emb = TransformerEmbedding(trg_voc_size, d_model, max_len, trg_pad_idx, drop_prob, device)
        self.decode = Decoder(n_layers, d_model, ffn_hidden, n_head, drop_prob)
        self.final_output = nn.Linear(d_model, trg_voc_size)
        self.device = device

    def forward(self, src, trg):
        """

        :param src: [batch_size, length]
        :param trg: [batch_size, length]
        :return:
        """
        src_mask = self.make_pad_mask(src, src)
        src_trg_mask = self.make_pad_mask(trg, src)
        trg_mask = self.make_pad_mask(trg, trg) * self.make_no_peak_mask(trg, trg)

        src = self.src_emb(src)
        env_src = self.encoder(src, src_mask)
        trg = self.trg_emb(trg)
        trg_output = self.decode(trg, env_src, trg_mask, src_trg_mask)
        output = self.final_output(trg_output)
        return output

    def make_pad_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        # batch_size, 1, 1, len_k
        k = k.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # batch_size, 1, len_q, len_k
        k = k.repeat(1, 1, len_q, 1)
        # batch_size, 1, len_q, 1
        q = q.ne(self.src_pad_idx).unsqueeze(1).unsqueeze(3)
        # batch_size, 1, len_q, len_k
        q = q.repeat(1, 1, 1, len_k)
        mask = (k & q).to(self.device)
        return mask

    def make_no_peak_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)

        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)

        return mask


if __name__ == '__main__':
    src_sample = torch.from_numpy(np.array([2, 12, 3, 4, 5, 1, 1, 1]))
    src_sample = src_sample.unsqueeze(0).repeat(5, 1)
    trg_sample = torch.from_numpy(np.array([2, 3, 3, 4, 5, 1, 1, 1]))
    trg_sample = trg_sample.unsqueeze(0).repeat(5, 1)
    model = Transformer(1, 1, 2, 20, 20, 128, 4, 10, 128, 2, 0.1)
    out = model(src_sample, trg_sample)
    writer = SummaryWriter()
    writer.add_graph(model, (src_sample, trg_sample))
    writer.flush()
