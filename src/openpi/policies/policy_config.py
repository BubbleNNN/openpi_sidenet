import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp
import safetensors.torch

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.policies.samsung_policy as samsung_policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


# FIX: policy inference needs to recognize the split SideNet checkpoint layout
# instead of only the legacy `model.safetensors` convention.
def _has_split_sidenet_checkpoint(checkpoint_dir: pathlib.Path) -> bool:
    return any(
        [
            (checkpoint_dir / "sidenet.safetensors").exists(),
            (checkpoint_dir / "sidenet_branches").exists(),
            (checkpoint_dir / "shared_parts.safetensors").exists(),
            (checkpoint_dir / "base_pi05.safetensors").exists(),
        ]
    )


def _resolve_sidenet_config_path(train_config: _config.TrainConfig) -> pathlib.Path:
    if train_config.sidenet.config_path is not None:
        return pathlib.Path(train_config.sidenet.config_path).expanduser().resolve()
    # FIX: mirror the training-side default when no explicit SideNet yaml path
    # is configured.
    return pathlib.Path(__file__).resolve().parents[3] / "sidenet" / "sidenet_config.yaml"


def _resolve_base_pi05_weight_path(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
) -> pathlib.Path:
    checkpoint_base_path = checkpoint_dir / "base_pi05.safetensors"
    if checkpoint_base_path.exists():
        return checkpoint_base_path
    if train_config.pytorch_weight_path is None:
        raise ValueError(
            "SideNet policy loading requires either `base_pi05.safetensors` in the "
            "checkpoint directory or `train_config.pytorch_weight_path`."
        )
    base_weight_dir = download.maybe_download(str(train_config.pytorch_weight_path))
    base_weight_path = base_weight_dir / "model.safetensors"
    if not base_weight_path.exists():
        raise FileNotFoundError(f"Base pi05 checkpoint not found at {base_weight_path}")
    return base_weight_path


def _shared_sidenet_state_dict(model) -> dict[str, Any]:
    shared_state: dict[str, Any] = {}
    for key, value in model.sidenet.state_dict().items():
        if (
            key == "fusion_vectors"
            or key.startswith("fusion.")
            or key.startswith("injectors.")
            or key.startswith("modality_gates.")
        ):
            shared_state[key] = value
    return shared_state


def _load_shared_parts_checkpoint(model, path: pathlib.Path, device: str) -> None:
    shared_state = safetensors.torch.load_file(str(path), device=device)
    model_shared_keys = set(_shared_sidenet_state_dict(model).keys())
    checkpoint_keys = set(shared_state.keys())
    unexpected_keys = sorted(checkpoint_keys - model_shared_keys)
    missing_keys = sorted(model_shared_keys - checkpoint_keys)
    if unexpected_keys:
        raise AssertionError(
            "Shared-parts checkpoint contains keys that are not shared SideNet parameters "
            f"for the current model: {unexpected_keys[:8]}"
        )
    if missing_keys:
        raise AssertionError(
            "Shared-parts checkpoint is missing shared SideNet parameters required by the current model: "
            f"{missing_keys[:8]}"
        )

    for key, loaded_value in shared_state.items():
        target_tensor = model.sidenet.state_dict()[key]
        if loaded_value.shape != target_tensor.shape:
            raise ValueError(
                f"Shared checkpoint tensor shape mismatch for `{key}`: "
                f"expected {tuple(target_tensor.shape)}, got {tuple(loaded_value.shape)}"
            )

    incompatible = model.sidenet.load_state_dict(shared_state, strict=False)
    if incompatible.unexpected_keys:
        raise AssertionError(f"Unexpected shared checkpoint keys after load: {incompatible.unexpected_keys}")
    remaining_missing = [key for key in incompatible.missing_keys if key in model_shared_keys]
    if remaining_missing:
        raise AssertionError(f"Missing shared checkpoint keys after load: {remaining_missing}")


def _load_split_sidenet_policy_model(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
    *,
    pytorch_device: str,
):
    # FIX: instantiate the SideNet wrapper for inference and load the split
    # checkpoint parts in the same order as training resume.
    from sidenet.pi05side import PI05withSideNet

    model = PI05withSideNet(
        sidenet_config_path=str(_resolve_sidenet_config_path(train_config)),
        pi05_config_name=train_config.name,
        pi05_weights_path=str(_resolve_base_pi05_weight_path(train_config, checkpoint_dir)),
    )

    strict = train_config.sidenet.checkpoint.strict_sidenet_load
    sidenet_path = checkpoint_dir / "sidenet.safetensors"
    if sidenet_path.exists():
        safetensors.torch.load_model(model.sidenet, str(sidenet_path), strict=strict, device=pytorch_device)

    branch_dir = checkpoint_dir / "sidenet_branches"
    if branch_dir.exists():
        for branch_path in sorted(branch_dir.glob("*.safetensors")):
            branch_name = branch_path.stem
            if branch_name not in model.sidenet.branches:
                raise ValueError(
                    f"Unknown SideNet branch `{branch_name}` in inference checkpoint load. "
                    f"Available branches: {sorted(model.sidenet.branches.keys())}"
                )
            safetensors.torch.load_model(
                model.sidenet.branches[branch_name],
                str(branch_path),
                strict=strict,
                device=pytorch_device,
            )

    shared_parts_path = checkpoint_dir / "shared_parts.safetensors"
    if shared_parts_path.exists():
        _load_shared_parts_checkpoint(model, shared_parts_path, pytorch_device)

    return model


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))
    checkpoint_dir = pathlib.Path(checkpoint_dir)

    # FIX: recognize both legacy PyTorch checkpoints and the new split SideNet
    # checkpoint layout.
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_split_sidenet = train_config.sidenet.enabled and _has_split_sidenet_checkpoint(checkpoint_dir)
    is_pytorch = os.path.exists(weight_path) or is_split_sidenet

    # Determine the device to use for PyTorch models before loading split
    # checkpoints so safetensors can materialize directly on the target device.
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    logging.info("Loading model...")
    if is_pytorch:
        if is_split_sidenet:
            # FIX: load the SideNet wrapper from the split checkpoint layout.
            model = _load_split_sidenet_policy_model(
                train_config,
                checkpoint_dir,
                pytorch_device=pytorch_device,
            )
            model.pi05_model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        else:
            model = train_config.model.load_pytorch(train_config, weight_path)
            model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    input_transforms = list(data_config.data_transforms.inputs)
    # FIX: when Samsung inference is configured to use an F/T window, swap in a
    # stateful transform that maintains the history locally and emits the same
    # `force_torque` key as training.
    if data_config.ft_window_size is not None and isinstance(train_config.data, _config.SamsungDataConfig):
        input_transforms = [
            samsung_policy.SamsungFTWindowInputs(
                model_type=transform.model_type,
                window_size=data_config.ft_window_size,
            )
            if isinstance(transform, samsung_policy.SamsungInputs)
            else transform
            for transform in input_transforms
        ]

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *input_transforms,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
