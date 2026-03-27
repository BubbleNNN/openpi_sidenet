from torch import nn


class MLPProjector(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0., layer_norm_eps=1e-5):
        super().__init__()
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.layers = nn.Sequential(
            nn.Linear(in_features, self.hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_features, self.out_features),
            nn.LayerNorm(self.out_features, eps=layer_norm_eps),
        )

    def forward(self, x):
        return self.layers(x)
