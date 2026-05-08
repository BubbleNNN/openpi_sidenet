"""Text-conditioned fixed-length readout for SideNet injection states.

JAX / Flax Linen equivalent of ``sidenet.fusion.text_conditioned_crossattn``.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class TextConditionedCrossAttention(nn.Module):
    """Generate fixed-length, text-conditioned injection hidden states.

    Data-flow:

    1. Project contextualized VLM text hidden states to ``d_sidenet`` with an
       MLP over normalised text states.
    2. Let learned layer-slot queries cross-attend the text tokens.
    3. Let the text-conditioned queries cross-attend modality tokens.
    4. Return ``(B, num_injected_layers, num_injection_tokens, d_sidenet)``.

    All branches use normal initialisation and are active from the start;
    SideNet identity is handled by the downstream LlamaAdapter gate instead of
    zero-initialising SideNet internals.
    """

    d_sidenet: int
    num_injected_layers: int
    num_injection_tokens: int
    vlm_hidden_dim: int
    num_heads: int

    @nn.compact
    def __call__(
        self,
        modality_tokens: jnp.ndarray,
        text_hidden_states: jnp.ndarray,
        text_mask: jnp.ndarray | None = None,
        modality_mask: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Return injection states with shape ``(B, L, K, d_sidenet)``."""
        if self.num_injected_layers <= 0 or self.num_injection_tokens <= 0:
            raise ValueError(
                "num_injected_layers and num_injection_tokens must both be positive, "
                f"got {self.num_injected_layers} and {self.num_injection_tokens}."
            )
        if modality_tokens.ndim != 3:
            raise ValueError(
                f"modality_tokens must have shape (B, S, d_sidenet), got {modality_tokens.shape}."
            )
        if text_hidden_states.ndim != 3:
            raise ValueError(
                "text_hidden_states must have shape (B, T, vlm_hidden_dim), "
                f"got {text_hidden_states.shape}."
            )
        if modality_tokens.shape[-1] != self.d_sidenet:
            raise ValueError(
                f"modality_tokens last dim {modality_tokens.shape[-1]} must match "
                f"d_sidenet {self.d_sidenet}."
            )
        if text_hidden_states.shape[-1] != self.vlm_hidden_dim:
            raise ValueError(
                f"text_hidden_states last dim {text_hidden_states.shape[-1]} must match "
                f"vlm_hidden_dim {self.vlm_hidden_dim}."
            )
        if modality_tokens.shape[0] != text_hidden_states.shape[0]:
            raise ValueError(
                "Batch size mismatch between modality_tokens "
                f"({modality_tokens.shape[0]}) and text_hidden_states "
                f"({text_hidden_states.shape[0]})."
            )

        batch_size = text_hidden_states.shape[0]
        num_readout_tokens = self.num_injected_layers * self.num_injection_tokens
        dtype = modality_tokens.dtype

        text_tokens = nn.LayerNorm(name="norm_text_input")(text_hidden_states)
        text_tokens = nn.Dense(self.d_sidenet, name="text_proj_fc1")(text_tokens)
        text_tokens = nn.gelu(text_tokens)
        text_tokens = nn.Dense(self.d_sidenet, name="text_proj_fc2")(text_tokens).astype(dtype)
        if text_mask is not None:
            if text_mask.shape != text_hidden_states.shape[:2]:
                raise ValueError(
                    f"text_mask shape {text_mask.shape} does not match text hidden state "
                    f"shape {text_hidden_states.shape[:2]}."
                )
            query_text_mask = jnp.ones((batch_size, num_readout_tokens), dtype=jnp.bool_)
            text_attn_mask = nn.make_attention_mask(query_text_mask, text_mask)
        else:
            text_attn_mask = None

        injection_queries = self.param(
            "injection_queries",
            nn.initializers.normal(stddev=0.02),
            (self.num_injected_layers, self.num_injection_tokens, self.d_sidenet),
        ).astype(dtype)
        q = jnp.broadcast_to(
            injection_queries[None, ...],
            (
                batch_size,
                self.num_injected_layers,
                self.num_injection_tokens,
                self.d_sidenet,
            ),
        )
        q = q.reshape(batch_size, num_readout_tokens, self.d_sidenet)

        text_tokens_normed = nn.LayerNorm(name="norm_text")(text_tokens)
        q_text = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            deterministic=deterministic,
            name="text_cross_attn",
        )(
            nn.LayerNorm(name="norm_q_text")(q),
            text_tokens_normed,
            text_tokens_normed,
            mask=text_attn_mask,
        )
        q = q + q_text

        if modality_mask is not None:
            if modality_mask.shape != modality_tokens.shape[:2]:
                raise ValueError(
                    f"modality_mask shape {modality_mask.shape} does not match modality "
                    f"token shape {modality_tokens.shape[:2]}."
                )
            query_modality_mask = jnp.ones((batch_size, num_readout_tokens), dtype=jnp.bool_)
            modality_attn_mask = nn.make_attention_mask(query_modality_mask, modality_mask)
        else:
            modality_attn_mask = None

        modality_context = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            deterministic=deterministic,
            name="modality_cross_attn",
        )(
            nn.LayerNorm(name="norm_q_modality")(q),
            nn.LayerNorm(name="norm_modality")(modality_tokens),
            modality_tokens,
            mask=modality_attn_mask,
        )
        out = q + modality_context

        residual = out
        out = nn.LayerNorm(name="norm_ffn")(out)
        out = nn.Dense(self.d_sidenet * 4, name="ffn_fc1")(out)
        out = nn.gelu(out)
        out = nn.Dense(self.d_sidenet, name="ffn_fc2")(out)
        out = residual + out

        return out.reshape(
            batch_size,
            self.num_injected_layers,
            self.num_injection_tokens,
            self.d_sidenet,
        )
