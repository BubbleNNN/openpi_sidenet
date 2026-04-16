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
    num_input_tokens: int = 4
    num_layers: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            x: ``(B, T, input_dim)`` or ``(B, input_dim)``.

        Returns:
            ``(B, num_queries, d_model)``
        """
        if self.num_input_tokens <= 0:
            raise ValueError(f"num_input_tokens must be positive, got {self.num_input_tokens}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")

        if x.ndim == 2:
            x = x[:, None, :]  # (B, 1, input_dim)

        if x.shape[1] == 1:
            kv = nn.Dense(self.d_model, name="single_frame_fc1")(x[:, 0])
            kv = nn.gelu(kv)
            kv = nn.Dense(
                self.num_input_tokens * self.d_model,
                name="single_frame_fc2",
            )(kv)
            kv = kv.reshape(x.shape[0], self.num_input_tokens, self.d_model)
        else:
            kv = nn.Dense(self.d_model, name="input_proj")(x)  # (B, T, d_model)

        B = kv.shape[0]
        latents = self.param(
            "latent_queries",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_queries, self.d_model),
        )
        latents = jnp.broadcast_to(latents, (B, self.num_queries, self.d_model))

        for layer_idx in range(self.num_layers):
            kv_normed = nn.LayerNorm(name=f"norm_cross_kv_{layer_idx}")(kv)
            q = nn.LayerNorm(name=f"norm_cross_q_{layer_idx}")(latents)
            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                deterministic=deterministic,
                name=f"cross_attn_{layer_idx}",
            )(q, kv_normed)
            latents = latents + attn_out

            latent_normed = nn.LayerNorm(name=f"norm_self_{layer_idx}")(latents)
            self_attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                deterministic=deterministic,
                name=f"self_attn_{layer_idx}",
            )(latent_normed, latent_normed)
            latents = latents + self_attn_out

            residual = latents
            out = nn.LayerNorm(name=f"norm_ffn_{layer_idx}")(latents)
            out = nn.Dense(self.d_model * 4, name=f"ffn_fc1_{layer_idx}")(out)
            out = nn.gelu(out)
            out = nn.Dense(self.d_model, name=f"ffn_fc2_{layer_idx}")(out)
            latents = residual + out

        return latents
