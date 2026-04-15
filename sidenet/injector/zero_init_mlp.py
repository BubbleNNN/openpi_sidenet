"""Zero-initialized MLP for safe injection into pretrained models.

The last linear layer is zero-initialized so that the injection path
starts at zero, preserving the pretrained model's behaviour at the
beginning of training.
"""

from torch import nn


class ZeroInitMLP(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.norm = nn.LayerNorm(out_features, eps=layer_norm_eps)

        # Zero-init last linear → injection starts at zero.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x
