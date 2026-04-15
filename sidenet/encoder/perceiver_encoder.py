"""Perceiver-style encoder with learnable query tokens and cross-attention.

Each branch uses this encoder to compress variable-length modality inputs
into a fixed number of output tokens whose dimension equals ``d_model``.
"""

import torch
from torch import nn


class PerceiverEncoder(nn.Module):
    """Per-modality perceiver encoder.

    Projects raw modality input to *d_model*, then uses learnable query
    tokens to cross-attend over the projected input.  The output has a
    fixed sequence length equal to *num_queries* regardless of input length.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_queries: int,
        num_heads: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.latent_queries = nn.Parameter(
            torch.randn(1, num_queries, d_model) * 0.02
        )

        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_kv = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )

        self.norm_ffn = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Modality input — ``(B, T, input_dim)`` or ``(B, input_dim)``.

        Returns:
            ``(B, num_queries, d_model)``
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (B, 1, input_dim)

        # Align dtype / device to the projection layer.
        x = x.to(device=self.input_proj.weight.device, dtype=self.input_proj.weight.dtype)

        kv = self.input_proj(x)  # (B, T, d_model)

        B = kv.shape[0]
        queries = self.latent_queries.expand(B, -1, -1)  # (B, num_queries, d_model)

        # Cross-attention: Q = learnable queries, K/V = projected input
        q = self.norm_q(queries)
        kv_normed = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q, kv_normed, kv_normed, need_weights=False)
        out = queries + attn_out

        # FFN
        out = out + self.ffn(self.norm_ffn(out))
        return out
