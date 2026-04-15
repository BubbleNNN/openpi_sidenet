"""Text-conditioned cross-attention for producing modality-fused hidden states.

JAX / Flax Linen equivalent of ``sidenet.fusion.text_conditioned_crossattn``.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class TextConditionedCrossAttention(nn.Module):
    """Cross-attention whose query is a learnable vector + projected text features."""

    d_model: int
    num_queries: int
    text_embed_dim: int
    num_heads: int

    @nn.compact
    def __call__(
        self,
        modality_tokens: jnp.ndarray,
        text_embeddings: jnp.ndarray,
        text_mask: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            modality_tokens: ``(B, N_mod, d_model)`` from concat + self-attn.
            text_embeddings: ``(B, L, text_embed_dim)`` contextualized VLM text states.
            text_mask: ``(B, L)`` bool — True for valid tokens.

        Returns:
            ``(B, num_queries, d_model)``
        """
        B = modality_tokens.shape[0]

        learnable_vectors = self.param(
            "learnable_vectors",
            nn.initializers.zeros,
            (1, self.num_queries, self.d_model),
        )

        # Project contextualized text features and mean-pool them.
        text_proj = nn.Dense(self.d_model, name="text_proj_fc1")(text_embeddings)
        text_proj = nn.gelu(text_proj)
        text_proj = nn.Dense(self.d_model, name="text_proj_fc2")(text_proj)

        if text_mask is not None:
            mask_f = text_mask[..., None].astype(text_proj.dtype)  # (B, L, 1)
            text_pooled = (text_proj * mask_f).sum(axis=1) / jnp.maximum(
                mask_f.sum(axis=1), 1.0
            )
        else:
            text_pooled = text_proj.mean(axis=1)
        # text_pooled: (B, d_model)

        # Query = learnable vectors + pooled text
        queries = (
            jnp.broadcast_to(learnable_vectors, (B, self.num_queries, self.d_model))
            + text_pooled[:, None, :]
        )

        # Cross-attention
        q = nn.LayerNorm(name="norm_q")(queries)
        kv = nn.LayerNorm(name="norm_kv")(modality_tokens)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            deterministic=deterministic,
            name="cross_attn",
        )(q, kv)
        out = queries + attn_out

        # FFN
        residual = out
        out = nn.LayerNorm(name="norm_ffn")(out)
        out = nn.Dense(self.d_model * 4, name="ffn_fc1")(out)
        out = nn.gelu(out)
        out = nn.Dense(self.d_model, name="ffn_fc2")(out)
        out = residual + out

        return out
