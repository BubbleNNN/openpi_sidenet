"""
Compose the side-network pipeline from configuration.

Current pipeline:
    per-branch: encoder -> projector -> refiner
    shared: fusion -> injector(s)

This module is intentionally split into two levels:
    - `SideNetBranch` handles a single modality branch
    - `SideNet` manages multiple branches, shared fusion, and injectors
"""

from __future__ import annotations

import torch
from torch import nn

from .encoder.cnn_encoder import CNNEncoder
from .fusion.crossattn_fusion import CrossAttnFusion
from .injector.mlp_injector import MLPInjector
from .parse_config import BranchConfig, ModuleConfig, SideNetConfig, load_sidenet_config
from .projector.mlp_projector import MLPProjector
from .refiner.attn_refiner import SelfAttnRefiner


ENCODER_REGISTRY: dict[str, type[nn.Module]] = {
    "cnn_encoder": CNNEncoder,
}

PROJECTOR_REGISTRY: dict[str, type[nn.Module]] = {
    "mlp_projector": MLPProjector,
}

REFINER_REGISTRY: dict[str, type[nn.Module]] = {
    "attn_refiner": SelfAttnRefiner,
}

FUSION_REGISTRY: dict[str, type[nn.Module]] = {
    "crossattn_fusion": CrossAttnFusion,
}

INJECTOR_REGISTRY: dict[str, type[nn.Module]] = {
    "mlp_injector": MLPInjector,
}


class SideNetBranch(nn.Module):
    """A single modality branch: encoder -> projector -> refiner."""

    def __init__(self, branch_cfg: BranchConfig):
        super().__init__()
        self.name = branch_cfg.name
        self.encoder = self._build_module(branch_cfg.encoder, ENCODER_REGISTRY, f"{self.name}.encoder")
        self.projector = self._build_module(branch_cfg.projector, PROJECTOR_REGISTRY, f"{self.name}.projector")
        self.refiner = self._build_module(branch_cfg.refiner, REFINER_REGISTRY, f"{self.name}.refiner")

    @staticmethod
    def _build_module(
        module_cfg: ModuleConfig,
        registry: dict[str, type[nn.Module]],
        module_name: str,
    ) -> nn.Module:
        if module_cfg.type not in registry:
            raise KeyError(
                f"Unknown {module_name} type `{module_cfg.type}`. "
                f"Available choices: {sorted(registry.keys())}"
            )
        module_cls = registry[module_cfg.type]
        return module_cls(**module_cfg.params)

    def forward(self, modality_input: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run one modality branch and keep intermediate outputs for debugging."""
        encoded_tokens = self.encoder(modality_input)
        projected_tokens = self.projector(encoded_tokens)
        refined_tokens = self.refiner(projected_tokens)

        return {
            "encoded_tokens": encoded_tokens,
            "projected_tokens": projected_tokens,
            "refined_tokens": refined_tokens,
        }


class SideNet(nn.Module):
    """Compose all currently implemented side-network components."""

    def __init__(self, config_or_path: str | SideNetConfig):
        super().__init__()

        if isinstance(config_or_path, SideNetConfig):
            self.config = config_or_path
        else:
            self.config = load_sidenet_config(config_or_path)

        self.branches = nn.ModuleDict(
            {
                branch_name: SideNetBranch(branch_cfg)
                for branch_name, branch_cfg in self.config.branches.items()
            }
        )
        # One learnable scalar gate per modality branch. They default to zero so
        # the attention write path in fusion starts from a no-op state unless an
        # external override gate is provided.
        self.modality_gates = nn.ParameterDict(
            {
                branch_name: nn.Parameter(torch.zeros(1))
                for branch_name in self.config.branches
            }
        )

        self.fusion = self._build_module(self.config.fusion, FUSION_REGISTRY, "fusion")
        self.num_fusion_vectors = self.config.fusion.params.get("num_fusion_vectors", 8)
        self.fusion_feature_dim = self.config.fusion.params["in_features"]

        if not isinstance(self.num_fusion_vectors, int) or self.num_fusion_vectors <= 0:
            raise ValueError("`fusion.params.num_fusion_vectors` must be a positive integer")

        # Shared learnable latents used as the fusion queries for all modalities.
        # They are stored with batch dimension 1 and expanded on demand at runtime.
        self.fusion_vectors = nn.Parameter(
            torch.zeros(1, self.num_fusion_vectors, self.fusion_feature_dim)
        )

        self.injectors = nn.ModuleDict(
            {
                name: self._build_module(injector_cfg.module, INJECTOR_REGISTRY, f"injector[{name}]")
                for name, injector_cfg in self.config.injectors.items()
            }
        )

    def _build_module(
        self,
        module_cfg: ModuleConfig,
        registry: dict[str, type[nn.Module]],
        module_name: str,
    ) -> nn.Module:
        if module_cfg.type not in registry:
            raise KeyError(
                f"Unknown {module_name} type `{module_cfg.type}`. "
                f"Available choices: {sorted(registry.keys())}"
            )

        module_cls = registry[module_cfg.type]
        module_params = dict(module_cfg.params)

        if module_name == "fusion":
            # `num_fusion_vectors` configures the internal learnable latents owned
            # by SideNet, not the fusion module constructor itself.
            module_params.pop("num_fusion_vectors", None)

        return module_cls(**module_params)

    def prepare_injections(self, fused_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Prepare injector outputs for downstream model integration.

        The output format depends on injector position:
            - `cross_attention`: keep the token sequence
            - `adarmsnorm`: pool tokens into a single condition vector
        """
        injections: dict[str, torch.Tensor] = {}

        for name, injector in self.injectors.items():
            injector_cfg = self.config.injectors[name]

            if injector_cfg.position == "cross_attention":
                injector_input = fused_tokens
            elif injector_cfg.position == "adarmsnorm":
                # AdaRMSNorm expects a single condition vector per sample.
                injector_input = fused_tokens.mean(dim=1)
            else:
                raise ValueError(
                    f"Unsupported injector position `{injector_cfg.position}` "
                    f"for injector `{name}`"
                )

            injections[name] = injector(injector_input)

        return injections

    def _get_fusion_vectors(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Expand the shared learnable fusion vectors for the current batch."""
        return self.fusion_vectors.to(device=device, dtype=dtype).expand(batch_size, -1, -1)

    def forward(
        self,
        modality_inputs: dict[str, torch.Tensor],
        modality_masks: dict[str, torch.Tensor] | None = None,
        gates: dict[str, torch.Tensor | float] | None = None,
        return_intermediates: bool = True,
    ) -> dict[str, object] | torch.Tensor:
        """Run the complete multi-branch side-network pipeline.

        Parameters
        ----------
        modality_inputs:
            Raw modality inputs keyed by branch name.
        modality_masks:
            Optional padding masks keyed by branch name.
        gates:
            Optional modality gates keyed by branch name. If omitted, SideNet
            uses its own learnable per-branch scalar gates.
        return_intermediates:
            If True, return branch-wise intermediate outputs together with fusion
            and injector outputs. If False, return fused tokens only.
        """
        if not modality_inputs:
            raise ValueError("`modality_inputs` must contain at least one modality")

        branch_outputs: dict[str, dict[str, torch.Tensor]] = {}
        modality_vectors: list[torch.Tensor] = []
        modality_masks_list: list[torch.Tensor | None] = []
        gates_list: list[torch.Tensor | float | None] = []

        for branch_name, modality_input in modality_inputs.items():
            if branch_name not in self.branches:
                raise KeyError(
                    f"Unknown branch `{branch_name}`. "
                    f"Available branches: {sorted(self.branches.keys())}"
                )

            branch_output = self.branches[branch_name](modality_input)
            branch_outputs[branch_name] = branch_output
            modality_vectors.append(branch_output["refined_tokens"])

            if modality_masks is None:
                modality_masks_list.append(None)
            else:
                modality_masks_list.append(modality_masks.get(branch_name))
            # flexible gating. Can either learn or pass from outside(e.g. when a certain modality is missing)
            if gates is None:
                gates_list.append(self.modality_gates[branch_name])
            else:
                gates_list.append(gates.get(branch_name, self.modality_gates[branch_name]))

        first_branch_name = next(iter(branch_outputs))
        first_refined_tokens = branch_outputs[first_branch_name]["refined_tokens"]
        fusion_vectors = self._get_fusion_vectors(
            batch_size=first_refined_tokens.shape[0],
            device=first_refined_tokens.device,
            dtype=first_refined_tokens.dtype,
        )

        fused_tokens = self.fusion(
            fusion_vectors=fusion_vectors,
            modality_vectors=modality_vectors,
            modality_masks=modality_masks_list,
            gates=gates_list,
        )

        if not return_intermediates:
            return fused_tokens

        injections = self.prepare_injections(fused_tokens)
        return {
            "branch_outputs": branch_outputs,
            "fused_tokens": fused_tokens,
            "injections": injections,
        }
