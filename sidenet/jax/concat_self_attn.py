"""Concatenate-and-self-attend module for multi-branch fusion.

JAX / Flax Linen equivalent of ``sidenet.fusion.concat_self_attn``.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class ConcatSelfAttention(nn.Module):
    """Self-attention over concatenated multi-branch tokens (Flax Linen)."""

    d_model: int
    num_heads: int

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            tokens: ``(B, N, d_model)`` concatenated from all branches.

        Returns:
            ``(B, N, d_model)``
        """
        residual = tokens
        normed = nn.LayerNorm(name="norm1")(tokens)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            deterministic=deterministic,
            name="self_attn",
        )(normed, normed)
        tokens = residual + attn_out

        residual = tokens
        out = nn.LayerNorm(name="norm2")(tokens)
        out = nn.Dense(self.d_model * 4, name="ffn_fc1")(out)
        out = nn.gelu(out)
        out = nn.Dense(self.d_model, name="ffn_fc2")(out)
        tokens = residual + out

        return tokens
