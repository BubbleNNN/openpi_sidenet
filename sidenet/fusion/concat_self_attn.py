"""Concatenate-and-self-attend module for multi-branch fusion.

All perceiver encoder outputs are concatenated along the sequence dimension
and then refined with a single self-attention + FFN block so that
information can flow across modality boundaries.
"""

import torch
from torch import nn


class ConcatSelfAttention(nn.Module):
    """Self-attention over concatenated multi-branch tokens."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: ``(B, N, d_model)`` concatenated from all branches.

        Returns:
            ``(B, N, d_model)``
        """
        normed = self.norm1(tokens)
        attn_out, _ = self.self_attn(normed, normed, normed, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens
