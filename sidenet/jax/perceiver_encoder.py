"""Perceiver-style encoder with learnable query tokens and cross-attention.

JAX / Flax Linen equivalent of ``sidenet.encoder.perceiver_encoder``.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class PerceiverEncoder(nn.Module):
    """Per-modality perceiver encoder (Flax Linen).

    Projects raw modality input to *d_model*, then uses learnable query
    tokens to cross-attend over the projected input.
    """

    input_dim: int
    d_model: int
    num_queries: int
    num_heads: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            x: ``(B, T, input_dim)`` or ``(B, input_dim)``.

        Returns:
            ``(B, num_queries, d_model)``
        """
        if x.ndim == 2:
            x = x[:, None, :]  # (B, 1, input_dim)

        kv = nn.Dense(self.d_model, name="input_proj")(x)  # (B, T, d_model)

        B = kv.shape[0]
        queries = self.param(
            "latent_queries",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_queries, self.d_model),
        )
        queries = jnp.broadcast_to(queries, (B, self.num_queries, self.d_model))

        # Cross-attention: Q = learnable queries, K/V = projected input
        q = nn.LayerNorm(name="norm_q")(queries)
        kv_normed = nn.LayerNorm(name="norm_kv")(kv)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            deterministic=deterministic,
            name="cross_attn",
        )(q, kv_normed)
        out = queries + attn_out

        # FFN
        residual = out
        out = nn.LayerNorm(name="norm_ffn")(out)
        out = nn.Dense(self.d_model * 4, name="ffn_fc1")(out)
        out = nn.gelu(out)
        out = nn.Dense(self.d_model, name="ffn_fc2")(out)
        out = residual + out

        return out
