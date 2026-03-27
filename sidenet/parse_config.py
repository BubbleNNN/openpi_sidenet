'''
This file contains a parser to translate the yaml config
into a python object flexibly
'''

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml 


@dataclass
class ModuleConfig:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class InjectorConfig:
    name: str
    position: str
    module: ModuleConfig


@dataclass
class BranchConfig:
    name: str
    encoder: ModuleConfig
    projector: ModuleConfig
    refiner: ModuleConfig
    
@dataclass
class SideNetConfig:
    fusion: ModuleConfig
    branches: dict[str, BranchConfig] = field(default_factory=dict)
    injectors: dict[str, InjectorConfig] = field(default_factory=dict)

    
def _parse_module_block(raw: dict[str, Any], name: str)->ModuleConfig:
    '''
    Parse blocks like:

    encoder:
      type: cnn_encoder
      params:
        in_channels: 6
        mid_channels: 64
        out_channels: 128

    into:
      ModuleConfig(type="cnn_encoder", params={...})
    '''
    
    if not isinstance(raw, dict):
        raise ValueError(f"Module block for {name} must be a dictionary")
    if 'type' not in raw:
        raise ValueError(f"Module block for {name} must contain a 'type' field")
    module_type = raw['type']
    if not isinstance(module_type, str):
        raise TypeError(f"Module block for {name} must have a string 'type' field")

    module_params = raw.get('params', {})
    if not isinstance(module_params, dict):
        raise TypeError(f"Module block for {name} must have a dictionary 'params' field")

    return ModuleConfig(type=module_type, params=module_params)

def _parse_injector_block(raw: dict[str, Any], name: str)->InjectorConfig:
    '''
    Parse blocks like:

    backbone:
      position: cross_attention
      type: mlp_injector
      params:
        in_features: 512
        mid_features: 1024
        out_features: 2048
    '''
    if not isinstance(raw, dict):
        raise ValueError(f"Injector block for {name} must be a dictionary")
    if raw.get('position') is None:
        raise ValueError(f"Injector block for {name} must contain a 'position' field")
    injector_pos = raw['position']
    if not isinstance(injector_pos, str):
        raise TypeError(f"Injector block for {name} must have a string 'position' field")

    if raw.get('type') is None:
        raise ValueError(f"Injector block for {name} must contain a 'type' field")
    injector_type = raw['type']
    if not isinstance(injector_type, str):
        raise TypeError(f"Injector block for {name} must have a string 'type' field")

    injector_params = raw.get('params', {})
    if not isinstance(injector_params, dict):
        raise TypeError(f"Injector block for {name} must have a dictionary 'params' field")

    return InjectorConfig(
        name=name,
        position=injector_pos,
        module=ModuleConfig(type=injector_type, params=injector_params),
    )


def _parse_branch_block(raw: dict[str, Any], name: str) -> BranchConfig:
    '''
    Parse blocks like:

    force_torque:
      encoder:
        type: cnn_encoder
        params: {...}
      projector:
        type: mlp_projector
        params: {...}
      refiner:
        type: attn_refiner
        params: {...}
    '''
    if not isinstance(raw, dict):
        raise ValueError(f"Branch block for {name} must be a dictionary")

    required_keys = ["encoder", "projector", "refiner"]
    for key in required_keys:
        if key not in raw:
            raise ValueError(f"Branch block for {name} must contain `{key}`")

    return BranchConfig(
        name=name,
        encoder=_parse_module_block(raw["encoder"], f"{name}.encoder"),
        projector=_parse_module_block(raw["projector"], f"{name}.projector"),
        refiner=_parse_module_block(raw["refiner"], f"{name}.refiner"),
    )

def load_sidenet_config(config_path: str) -> SideNetConfig:
    '''
    Load the yaml config file and parse it into a SideNetConfig object.
    '''
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    if not isinstance(raw_config, dict):
        raise ValueError("Top-level config must be a dictionary")
        
    required_top_keys = ["branches", "fusion"]
    for key in required_top_keys:
        if key not in raw_config:
            raise ValueError(f"Missing required top-level key: `{key}`")    

    raw_branches = raw_config["branches"]
    if not isinstance(raw_branches, dict):
        raise ValueError("`branches` must be a dictionary")
    if not raw_branches:
        raise ValueError("`branches` must contain at least one branch")

    branches_cfg = {
        branch_name: _parse_branch_block(branch_raw, branch_name)
        for branch_name, branch_raw in raw_branches.items()
    }

    fusion_cfg = _parse_module_block(raw_config["fusion"], "fusion")

    raw_injectors = raw_config.get("injector", {})
    if not isinstance(raw_injectors, dict):
        raise ValueError("`injector` must be a dictionary")

    injectors = {
        injector_name: _parse_injector_block(injector_raw, injector_name)
        for injector_name, injector_raw in raw_injectors.items()
    }
    return SideNetConfig(
        branches=branches_cfg,
        fusion=fusion_cfg,
        injectors=injectors,
    )
