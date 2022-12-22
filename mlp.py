from torch import nn


class MLP(nn.Module):
    def __init__(self, d_model, hidden, drop_prob):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden, d_model)
        )

    def forward(self, x):
        return self.mlp(x)
