"""
Parse the SideNet v2 YAML config into typed Python objects.

The v2 config is flat and describes the perceiver-based architecture:
    d_model, num_perceiver_queries, num_perceiver_layers, num_input_tokens,
    num_fusion_queries, num_heads, text_embed_dim, branches
    (each with input_dim), output_mlp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class BranchConfig:
    """Config for a single perceiver-encoder branch."""
    input_dim: int


@dataclass
class SideNetConfig:
    """Top-level SideNet v2 configuration."""
    d_model: int
    num_perceiver_queries: int
    num_perceiver_layers: int
    num_input_tokens: int
    num_fusion_queries: int
    num_heads: int
    text_embed_dim: int
    output_hidden_features: int
    output_dim: int
    branches: dict[str, BranchConfig] = field(default_factory=dict)


def load_sidenet_config(config_path: str) -> SideNetConfig:
    """Load a SideNet v2 YAML config and return a typed ``SideNetConfig``."""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be a dictionary")

    required_keys = ["d_model", "num_perceiver_queries", "num_fusion_queries",
                     "num_heads", "text_embed_dim", "branches", "output_mlp"]
    for key in required_keys:
        if key not in raw:
            raise ValueError(f"Missing required top-level key: `{key}`")

    # --- branches ---
    raw_branches = raw["branches"]
    if not isinstance(raw_branches, dict) or not raw_branches:
        raise ValueError("`branches` must be a non-empty dictionary")

    branches: dict[str, BranchConfig] = {}
    for name, branch_raw in raw_branches.items():
        if not isinstance(branch_raw, dict) or "input_dim" not in branch_raw:
            raise ValueError(
                f"Branch `{name}` must be a dict with at least `input_dim`"
            )
        branches[name] = BranchConfig(input_dim=int(branch_raw["input_dim"]))

    # --- output MLP ---
    raw_mlp = raw["output_mlp"]
    if not isinstance(raw_mlp, dict):
        raise ValueError("`output_mlp` must be a dictionary")
    for k in ("hidden_features", "out_features"):
        if k not in raw_mlp:
            raise ValueError(f"`output_mlp` must contain `{k}`")

    return SideNetConfig(
        d_model=int(raw["d_model"]),
        num_perceiver_queries=int(raw["num_perceiver_queries"]),
        num_perceiver_layers=int(raw.get("num_perceiver_layers", 2)),
        num_input_tokens=int(raw.get("num_input_tokens", 4)),
        num_fusion_queries=int(raw["num_fusion_queries"]),
        num_heads=int(raw["num_heads"]),
        text_embed_dim=int(raw["text_embed_dim"]),
        output_hidden_features=int(raw_mlp["hidden_features"]),
        output_dim=int(raw_mlp["out_features"]),
        branches=branches,
    )
