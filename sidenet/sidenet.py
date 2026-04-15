"""
Compose the side-network pipeline from configuration.

Architecture (v2 — perceiver-based):
    per-branch : PerceiverEncoder  (learnable-query cross-attention)
    shared     : ConcatSelfAttention → TextConditionedCrossAttention → ZeroInitMLP

The final output is a fixed-length token sequence whose hidden dimension
matches the action-expert width, ready to be concatenated with suffix
embeddings before the action expert.
"""

from __future__ import annotations

import torch
from torch import nn

from .encoder.perceiver_encoder import PerceiverEncoder
from .fusion.concat_self_attn import ConcatSelfAttention
from .fusion.text_conditioned_crossattn import TextConditionedCrossAttention
from .injector.zero_init_mlp import ZeroInitMLP
from .parse_config import SideNetConfig, load_sidenet_config


class SideNet(nn.Module):
    """Perceiver-based multi-branch side-network.

    Data-flow
    ---------
    1. Each branch runs a ``PerceiverEncoder`` over its modality input.
    2. All branch outputs are concatenated and self-attended.
    3. A learnable query vector (combined with projected VLA text embeddings)
       cross-attends to the self-attended tokens.
    4. The result passes through a zero-initialised MLP whose output dimension
       matches the action-expert hidden width.
    """

    def __init__(self, config_or_path: str | SideNetConfig):
        super().__init__()

        if isinstance(config_or_path, SideNetConfig):
            self.config = config_or_path
        else:
            self.config = load_sidenet_config(config_or_path)

        d_model = self.config.d_model
        num_heads = self.config.num_heads

        # --- per-branch perceiver encoders ---
        self.branches = nn.ModuleDict(
            {
                name: PerceiverEncoder(
                    input_dim=branch_cfg.input_dim,
                    d_model=d_model,
                    num_queries=self.config.num_perceiver_queries,
                    num_heads=num_heads,
                )
                for name, branch_cfg in self.config.branches.items()
            }
        )

        # --- shared: concatenate + self-attention ---
        self.concat_self_attn = ConcatSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
        )

        # --- shared: text-conditioned cross-attention ---
        self.text_conditioned_crossattn = TextConditionedCrossAttention(
            d_model=d_model,
            num_queries=self.config.num_fusion_queries,
            text_embed_dim=self.config.text_embed_dim,
            num_heads=num_heads,
        )

        # --- zero-init output MLP ---
        self.output_mlp = ZeroInitMLP(
            in_features=d_model,
            hidden_features=self.config.output_hidden_features,
            out_features=self.config.output_dim,
        )

        self._initialize_weights()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _initialize_weights(self) -> None:
        """Branch modules get standard init; shared write-path stays near zero."""
        for branch in self.branches.values():
            self._init_standard(branch)
        self._init_standard(self.concat_self_attn)
        self._init_standard(self.text_conditioned_crossattn)
        # output_mlp already zero-inits its last linear in its own __init__.

    @staticmethod
    def _init_standard(module: nn.Module) -> None:
        for sub in module.modules():
            if isinstance(sub, nn.Linear):
                nn.init.xavier_uniform_(sub.weight)
                if sub.bias is not None:
                    nn.init.zeros_(sub.bias)
            elif isinstance(sub, nn.MultiheadAttention):
                nn.init.xavier_uniform_(sub.in_proj_weight)
                if sub.in_proj_bias is not None:
                    nn.init.zeros_(sub.in_proj_bias)
                nn.init.xavier_uniform_(sub.out_proj.weight)
                if sub.out_proj.bias is not None:
                    nn.init.zeros_(sub.out_proj.bias)
            elif isinstance(sub, nn.LayerNorm | nn.BatchNorm1d | nn.GroupNorm):
                if sub.weight is not None:
                    nn.init.ones_(sub.weight)
                if sub.bias is not None:
                    nn.init.zeros_(sub.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        modality_inputs: dict[str, torch.Tensor],
        text_embeddings: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the full side-network pipeline.

        Parameters
        ----------
        modality_inputs:
            Raw modality tensors keyed by branch name.
            Each value is ``(B, T, input_dim)`` or ``(B, input_dim)``.
        text_embeddings:
            Language token embeddings from the VLA backbone's embedding layer,
            shape ``(B, L, text_embed_dim)``.
        text_mask:
            Bool mask for *text_embeddings*, ``(B, L)``, True = valid token.

        Returns
        -------
        ``(B, num_fusion_queries, output_dim)``
            Tokens ready for concatenation with the action-expert suffix.
        """
        if not modality_inputs:
            raise ValueError("`modality_inputs` must contain at least one modality")

        # 1. Per-branch perceiver encoding
        branch_outputs: list[torch.Tensor] = []
        for name in sorted(self.branches.keys()):
            if name not in modality_inputs:
                raise KeyError(
                    f"Missing modality input for branch `{name}`. "
                    f"Available inputs: {sorted(modality_inputs.keys())}"
                )
            branch_out = self.branches[name](modality_inputs[name])
            branch_outputs.append(branch_out)  # (B, num_perceiver_queries, d_model)

        # 2. Concatenate + self-attention
        concat_tokens = torch.cat(branch_outputs, dim=1)  # (B, total_queries, d_model)
        fused_tokens = self.concat_self_attn(concat_tokens)

        # 3. Text-conditioned cross-attention
        output = self.text_conditioned_crossattn(
            fused_tokens,
            text_embeddings,
            text_mask,
        )  # (B, num_fusion_queries, d_model)

        # 4. Zero-init MLP → action-expert width
        output = self.output_mlp(output)  # (B, num_fusion_queries, output_dim)

        return output
