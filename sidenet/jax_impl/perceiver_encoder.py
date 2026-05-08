"""Perceiver-style encoder over physics tokens.

JAX / Flax Linen equivalent of ``sidenet.encoder.perceiver_encoder``.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class PerceiverEncoder(nn.Module):
    """Per-modality perceiver encoder (Flax Linen).

    Uses learnable query tokens to cross-attend over physics-aware modality
    tokens produced by ``PhysicsTokenizer``.
    """

    d_embedding: int
    d_sidenet: int
    num_encoding_tokens: int
    num_heads: int
    num_layers: int = 1

    @nn.compact
    def __call__(self, phy_tokens: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            phy_tokens: ``(B, L_phy, d_embedding)``.

        Returns:
            ``(B, num_encoding_tokens, d_sidenet)``
        """
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if phy_tokens.ndim != 3:
            raise ValueError(
                f"phy_tokens must have shape (B, L_phy, d_embedding), got {phy_tokens.shape}."
            )
        if phy_tokens.shape[-1] != self.d_embedding:
            raise ValueError(
                f"phy_tokens last dim {phy_tokens.shape[-1]} must match "
                f"d_embedding {self.d_embedding}."
            )
        if self.d_embedding != self.d_sidenet:
            phy_tokens = nn.Dense(self.d_sidenet, name="input_proj")(phy_tokens)

        B = phy_tokens.shape[0]
        latents = self.param(
            "latent_queries",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_encoding_tokens, self.d_sidenet),
        )
        latents = jnp.broadcast_to(
            latents, (B, self.num_encoding_tokens, self.d_sidenet)
        )

        for layer_idx in range(self.num_layers):
            kv_normed = nn.LayerNorm(name=f"norm_cross_kv_{layer_idx}")(phy_tokens)
            q = nn.LayerNorm(name=f"norm_cross_q_{layer_idx}")(latents)
            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                deterministic=deterministic,
                name=f"cross_attn_{layer_idx}",
            )(q, kv_normed, phy_tokens)
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
            out = nn.Dense(self.d_sidenet * 4, name=f"ffn_fc1_{layer_idx}")(out)
            out = nn.gelu(out)
            out = nn.Dense(self.d_sidenet, name=f"ffn_fc2_{layer_idx}")(out)
            latents = residual + out

        modality_embedding = self.param(
            "modality_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, self.num_encoding_tokens, self.d_sidenet),
        )
        return latents + modality_embedding.astype(latents.dtype)
