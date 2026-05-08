"""
SideNet v2 — JAX / Flax Linen implementation.

Architecture (perceiver + text-conditioned fixed readout):
    per-branch : PhysicsTokenizer → PerceiverEncoder
    shared     : concatenate branch tokens → injection-query cross-attention → output MLP

The normal ``__call__`` path maps fixed-length injection hidden states to the
action-expert width. This final SideNet output is what the adapter path injects.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp

from .perceiver_encoder import PerceiverEncoder
from .physics_tokenizer import PhysicsTokenizer
from .text_conditioned_crossattn import TextConditionedCrossAttention


class StandardOutputMLP(nn.Module):
    """Dense → GELU → Dense → LayerNorm with standard Linen initialisation."""

    hidden_features: int
    out_features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_features, name="fc1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.out_features, name="fc2")(x)
        x = nn.LayerNorm(name="norm")(x)
        return x


class SideNet(nn.Module):
    """Perceiver-based multi-branch side-network (Flax Linen).

    Data-flow
    ---------
    1. Each branch tokenizes raw modality input into physics-aware tokens.
    2. Each branch runs a ``PerceiverEncoder`` over those physics tokens.
    3. All branch outputs are concatenated without an extra self-attention stage.
    4. Learned layer-slot injection queries first cross-attend contextualized
       text hidden states, then cross-attend the modality tokens.
    5. The result passes through a shared output MLP whose output dimension
       matches the action-expert hidden width; these are the final multimodal
       hidden states used for injection.

    Usage
    -----
    Instantiate directly::

        model = SideNet(
            branch_input_dims=(("ft_sensor", 12),),
            d_embedding=512,
            d_sidenet=512,
            ...
        )

    Or from the shared YAML config::

        model = SideNet.from_yaml_config("sidenet/sidenet_config.yaml")
    """

    # Branch configs as a hashable tuple of (name, input_dim) pairs.
    branch_input_dims: tuple  # ((name, input_dim), ...)
    d_embedding: int = 512
    d_sidenet: int = 512
    num_encoding_tokens: int = 8
    num_injected_layers: int = 1
    num_injection_tokens: int = 8
    num_heads: int = 8
    num_perceiver_layers: int = 1
    vlm_hidden_dim: int = 2048
    output_hidden_features: int = 1024
    action_expert_hidden_dim: int = 1024

    def setup(self):
        self.tokenizers = {
            name: PhysicsTokenizer(
                modality_name=name,
                input_dim=input_dim,
                d_embedding=self.d_embedding,
                name=f"tokenizer_{name}",
            )
            for name, input_dim in self.branch_input_dims
        }
        self.branches = {
            name: PerceiverEncoder(
                d_embedding=self.d_embedding,
                d_sidenet=self.d_sidenet,
                num_encoding_tokens=self.num_encoding_tokens,
                num_layers=self.num_perceiver_layers,
                num_heads=self.num_heads,
                name=f"branch_{name}",
            )
            for name, input_dim in self.branch_input_dims
        }

        self.text_cross_attn = TextConditionedCrossAttention(
            d_sidenet=self.d_sidenet,
            num_injected_layers=self.num_injected_layers,
            num_injection_tokens=self.num_injection_tokens,
            vlm_hidden_dim=self.vlm_hidden_dim,
            num_heads=self.num_heads,
        )

        self.output_mlp = StandardOutputMLP(
            hidden_features=self.output_hidden_features,
            out_features=self.action_expert_hidden_dim,
        )

    def _forward_branches_and_fuse(
        self,
        modality_inputs: dict[str, jnp.ndarray],
        text_hidden_states: jnp.ndarray,
        text_mask: jnp.ndarray | None,
        deterministic: bool,
        modality_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Steps 1-4: physics tokenization → branch encoding → concat → cross-attn.

        Returns injection hidden states with shape
        ``(B, num_injected_layers, num_injection_tokens, d_sidenet)`` before the
        output MLP.
        """
        # 1-2. Per-branch physics tokenization + perceiver encoding
        branch_outputs = []
        for name, _ in sorted(self.branch_input_dims):
            if name not in modality_inputs:
                raise KeyError(
                    f"Missing modality input for branch `{name}`. "
                    f"Available inputs: {sorted(modality_inputs.keys())}"
                )

            branch_input = modality_inputs[name]
            phy_tokens = self.tokenizers[name](branch_input)
            out = self.branches[name](phy_tokens, deterministic=deterministic)
            branch_outputs.append(out)

        # 3. Concatenate branch tokens directly. The explicit concat self-attn
        # stage is intentionally removed; text-conditioned fusion happens through
        # learned injection queries attending to this modality memory.
        fused_tokens = jnp.concatenate(branch_outputs, axis=1)

        # 4. Text-conditioned cross-attention
        return self.text_cross_attn(
            fused_tokens,
            text_hidden_states,
            text_mask,
            modality_mask=modality_mask,
            deterministic=deterministic,
        )

    def compute_fused_memory(
        self,
        modality_inputs: dict[str, jnp.ndarray],
        text_hidden_states: jnp.ndarray,
        text_mask: jnp.ndarray | None = None,
        deterministic: bool = True,
        modality_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Return injection hidden states *before* the output MLP.

        Shape: ``(B, num_injected_layers, num_injection_tokens, d_sidenet)``.

        Use this when the caller has its own per-layer projection (e.g.
        the LlamaAdapter wrapper).
        """
        return self._forward_branches_and_fuse(
            modality_inputs, text_hidden_states, text_mask, deterministic, modality_mask,
        )

    def __call__(
        self,
        modality_inputs: dict[str, jnp.ndarray],
        text_hidden_states: jnp.ndarray,
        text_mask: jnp.ndarray | None = None,
        deterministic: bool = True,
        modality_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Run the full side-network pipeline.

        Parameters
        ----------
        modality_inputs:
            Raw modality tensors keyed by branch name.
            The current JAX wrapper only wires single-frame ``ft_sensor`` and
            expects ``(B, input_dim)`` or ``(B, 1, input_dim)`` for that branch.
        text_hidden_states:
            Contextualized VLM text hidden states, shape
            ``(B, L, vlm_hidden_dim)``.
        text_mask:
            Bool mask for *text_hidden_states*, ``(B, L)``, True = valid.
        deterministic:
            If True, disable dropout (inference mode).
        modality_mask:
            Optional bool mask for concatenated modality tokens, ``(B, S)``.
            The current fixed-branch JAX wrapper leaves this as ``None``.

        Returns
        -------
        ``(B, num_injected_layers, num_injection_tokens, action_expert_hidden_dim)``
        """
        output = self._forward_branches_and_fuse(
            modality_inputs, text_hidden_states, text_mask, deterministic, modality_mask,
        )
        # 4. Output MLP → action-expert width
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
            d_embedding=overrides.get("d_embedding", cfg.d_embedding),
            d_sidenet=overrides.get("d_sidenet", cfg.d_sidenet),
            num_encoding_tokens=overrides.get(
                "num_encoding_tokens", cfg.num_encoding_tokens,
            ),
            num_injected_layers=overrides.get("num_injected_layers", 1),
            num_injection_tokens=overrides.get("num_injection_tokens", cfg.num_injection_tokens),
            num_perceiver_layers=overrides.get("num_perceiver_layers", cfg.num_perceiver_layers),
            num_heads=overrides.get("num_heads", cfg.num_heads),
            vlm_hidden_dim=overrides.get("vlm_hidden_dim", cfg.vlm_hidden_dim),
            output_hidden_features=overrides.get("output_hidden_features", cfg.output_hidden_features),
            action_expert_hidden_dim=overrides.get(
                "action_expert_hidden_dim", cfg.action_expert_hidden_dim,
            ),
        )
