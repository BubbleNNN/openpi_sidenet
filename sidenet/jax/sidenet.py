"""
SideNet v2 — JAX / Flax Linen implementation.

Architecture (perceiver-based):
    per-branch : PerceiverEncoder  (learnable-query cross-attention)
    shared     : ConcatSelfAttention → TextConditionedCrossAttention → ZeroInitMLP

The final output is a fixed-length token sequence whose hidden dimension
matches the action-expert width, ready to be concatenated with suffix
embeddings before the action expert.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from .perceiver_encoder import PerceiverEncoder
from .concat_self_attn import ConcatSelfAttention
from .text_conditioned_crossattn import TextConditionedCrossAttention
from .zero_init_mlp import ZeroInitMLP


class SideNet(nn.Module):
    """Perceiver-based multi-branch side-network (Flax Linen).

    Data-flow
    ---------
    1. Each branch runs a ``PerceiverEncoder`` over its modality input.
    2. All branch outputs are concatenated and self-attended.
    3. A learnable query vector (combined with projected contextualized VLM text
       features)
       cross-attends to the self-attended tokens.
    4. The result passes through a zero-initialised MLP whose output dimension
       matches the action-expert hidden width.

    Usage
    -----
    Instantiate directly::

        model = SideNet(
            branch_input_dims=(("ft_sensor", 12),),
            d_model=512,
            ...
        )

    Or from the shared YAML config::

        model = SideNet.from_yaml_config("sidenet/sidenet_config.yaml")
    """

    # Branch configs as a hashable tuple of (name, input_dim) pairs.
    branch_input_dims: tuple  # ((name, input_dim), ...)
    d_model: int = 512
    num_perceiver_queries: int = 8
    num_fusion_queries: int = 8
    num_heads: int = 8
    num_input_tokens: int = 4
    num_perceiver_layers: int = 2
    text_embed_dim: int = 2048
    output_hidden_features: int = 1024
    output_dim: int = 1024

    def setup(self):
        self.branches = {
            name: PerceiverEncoder(
                input_dim=input_dim,
                d_model=self.d_model,
                num_queries=self.num_perceiver_queries,
                num_input_tokens=self.num_input_tokens,
                num_layers=self.num_perceiver_layers,
                num_heads=self.num_heads,
                name=f"branch_{name}",
            )
            for name, input_dim in self.branch_input_dims
        }

        self.concat_self_attn = ConcatSelfAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
        )

        self.text_cross_attn = TextConditionedCrossAttention(
            d_model=self.d_model,
            num_queries=self.num_fusion_queries,
            text_embed_dim=self.text_embed_dim,
            num_heads=self.num_heads,
        )

        self.output_mlp = ZeroInitMLP(
            hidden_features=self.output_hidden_features,
            out_features=self.output_dim,
        )

    def __call__(
        self,
        modality_inputs: dict[str, jnp.ndarray],
        text_embeddings: jnp.ndarray,
        text_mask: jnp.ndarray | None = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Run the full side-network pipeline.

        Parameters
        ----------
        modality_inputs:
            Raw modality tensors keyed by branch name.
            The current JAX wrapper only wires single-frame ``ft_sensor`` and
            expects ``(B, input_dim)`` or ``(B, 1, input_dim)`` for that branch.
        text_embeddings:
            Contextualized VLM text hidden states, shape
            ``(B, L, text_embed_dim)``.
        text_mask:
            Bool mask for *text_embeddings*, ``(B, L)``, True = valid.
        deterministic:
            If True, disable dropout (inference mode).

        Returns
        -------
        ``(B, num_fusion_queries, output_dim)``
        """
        # 1. Per-branch perceiver encoding
        branch_outputs = []
        for name, _ in sorted(self.branch_input_dims):
            if name not in modality_inputs:
                raise KeyError(
                    f"Missing modality input for branch `{name}`. "
                    f"Available inputs: {sorted(modality_inputs.keys())}"
                )

            branch_input = modality_inputs[name]
            if name == "ft_sensor" and branch_input.ndim == 3 and branch_input.shape[1] != 1:
                raise ValueError(
                    "JAX SideNet currently expects single-frame `ft_sensor` input "
                    "with shape `(B, 12)` or `(B, 1, 12)`, not a temporal window."
                )

            out = self.branches[name](branch_input, deterministic=deterministic)
            branch_outputs.append(out)

        # 2. Concatenate + self-attention
        concat_tokens = jnp.concatenate(branch_outputs, axis=1)
        fused_tokens = self.concat_self_attn(concat_tokens, deterministic=deterministic)

        # 3. Text-conditioned cross-attention
        output = self.text_cross_attn(
            fused_tokens, text_embeddings, text_mask, deterministic=deterministic,
        )

        # 4. Zero-init MLP → action-expert width
        output = self.output_mlp(output)

        return output

    @classmethod
    def from_yaml_config(cls, config_path: str, **overrides) -> "SideNet":
        """Build a ``SideNet`` from the shared YAML configuration file.

        Any keyword override takes precedence over the YAML value.
        """
        from ..parse_config import load_sidenet_config

        cfg = load_sidenet_config(config_path)
        branch_input_dims = tuple(
            (name, bcfg.input_dim) for name, bcfg in sorted(cfg.branches.items())
        )
        return cls(
            branch_input_dims=branch_input_dims,
            d_model=overrides.get("d_model", cfg.d_model),
            num_perceiver_queries=overrides.get("num_perceiver_queries", cfg.num_perceiver_queries),
            num_input_tokens=overrides.get("num_input_tokens", cfg.num_input_tokens),
            num_perceiver_layers=overrides.get("num_perceiver_layers", cfg.num_perceiver_layers),
            num_fusion_queries=overrides.get("num_fusion_queries", cfg.num_fusion_queries),
            num_heads=overrides.get("num_heads", cfg.num_heads),
            text_embed_dim=overrides.get("text_embed_dim", cfg.text_embed_dim),
            output_hidden_features=overrides.get("output_hidden_features", cfg.output_hidden_features),
            output_dim=overrides.get("output_dim", cfg.output_dim),
        )
