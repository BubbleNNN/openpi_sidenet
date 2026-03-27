from torch import nn


class SelfAttnRefiner(nn.Module):
    def __init__(self, in_features, num_heads, dropout=0., layer_norm_eps=1e-5):
        super().__init__()
        self.attn = nn.MultiheadAttention(in_features, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(in_features, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(in_features, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, in_features * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features * 4, in_features),
        )

    def forward(self, x):
        # x shape: (batch_size, time_steps, features)
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x  # (batch_size, time_steps, features)
