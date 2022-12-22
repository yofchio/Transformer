import math

import torch

from einops import rearrange
from pytorch_model_summary import summary

from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_tensor = self.d_model // self.n_head
        self.scale = math.sqrt(self.d_tensor)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_contact = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)

        # [botchstz0, n heod, length, d tensor]
        k_t = k.transpose(2, 3)  # [botchsize, nL heod, d_ tensor, length]
        score = torch.matmul(q, k_t) / self.scale
        if mask is not None:
            score.masked_fill(mask == 0, -e)
        score = self.softmax(score)
        v = torch.matmul(score, v)  # (batchsize, n.heod, length, d.tensor]
        output = self.concat(v)
        output = self.w_contact(output)
        return output

    def split(self, tensor):
        output = rearrange(tensor, "b l (n d) -> b n l d", n=self.n_head)
        return output

    def concat(self, tensor):
        tensor = rearrange(tensor, "b n l d -> b l (n d)")
        return tensor


def main():
    model = MultiHeadAttention(128, 4)
    sample = torch.randn((5, 100, 128))
    out = model(sample, sample, sample)
    print(out.shape)

    summary(model, sample, sample, sample, print_summary=True, max_depth=2)
    #
    # writer = SummaryWriter()
    # writer.add_graph(model, sample)


if __name__ == '__main__':
    main()
