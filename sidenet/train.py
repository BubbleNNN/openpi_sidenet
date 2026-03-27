"""
This file is a slighlt modified version of the original train_pytorch.py.
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc
import logging
import os
import pathlib
import platform
import shutil
import time
from collections.abc import Mapping

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
# FIX: replace wandb with local TensorBoard logging for offline training.
from torch.utils.tensorboard import SummaryWriter

import openpi.models.pi0_config
import openpi.models.model as _model
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data
# FIX: import the SideNet-wrapped pi05 model instead of the base PI0Pytorch only.
from .pi05side import PI05withSideNet


# FIX: filter Observation.modalities against the configured SideNet branch list so
# training can use the standard `(Observation, actions)` loader contract.
def select_observation_modalities(
    observation: _model.Observation,
    modalities: tuple[str, ...] | list[str] | None = None,
) -> _model.Observation:
    available_modalities = observation.modalities or {}
    if modalities:
        missing_modalities = [key for key in modalities if key not in available_modalities]
        if missing_modalities:
            raise ValueError(
                "Configured SideNet modalities were not found in Observation.modalities. "
                f"Missing: {missing_modalities}; available: {sorted(available_modalities)}"
            )
        selected_modalities = {key: available_modalities[key] for key in modalities}
    else:
        selected_modalities = available_modalities

    if not selected_modalities:
        raise ValueError(
            "No SideNet modalities were found in Observation.modalities. "
            "Check the data transforms and `config.sidenet.modalities`."
        )

    return observation.replace(modalities=selected_modalities)


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


# FIX: TensorBoard initialization replaces the original wandb initialization.
def init_tensorboard(config: _config.TrainConfig, *, enabled: bool = True) -> SummaryWriter | None:
    """Initialize local TensorBoard logging.

    We intentionally keep using `config.wandb_enabled` as the boolean switch so
    existing training configs continue to work without additional config edits.
    """
    if not enabled:
        return None

    ckpt_dir = get_training_checkpoint_dir(config)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    log_dir = ckpt_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text("run/config", repr(dataclasses.asdict(config)), global_step=0)
    return writer


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def unwrap_model(model):
    """Remove the DDP wrapper when present."""
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


# FIX: allow SideNet training to optionally use a dedicated checkpoint root.
def get_training_checkpoint_dir(config: _config.TrainConfig) -> pathlib.Path:
    override_dir = config.sidenet.checkpoint.sidenet_checkpoint_dir
    if override_dir:
        return pathlib.Path(override_dir).expanduser().resolve()
    return config.checkpoint_dir


# FIX: configure which parts of the wrapper are trainable before optimizer creation.
def configure_trainable_parameters(model, config: _config.TrainConfig) -> None:
    unwrapped_model = unwrap_model(model)

    # pi05 trainability is controlled globally.
    for param in unwrapped_model.pi05_model.parameters():
        param.requires_grad_(config.sidenet.train_pi05)

    requested_branches = tuple(config.sidenet.trainable_branches)
    available_branches = tuple(unwrapped_model.sidenet.branches.keys())
    if requested_branches:
        unknown = sorted(set(requested_branches) - set(available_branches))
        if unknown:
            raise ValueError(
                f"Unknown trainable SideNet branches {unknown}. "
                f"Available branches: {sorted(available_branches)}"
            )

    for branch_name, branch in unwrapped_model.sidenet.branches.items():
        branch_is_trainable = not requested_branches or branch_name in requested_branches
        for param in branch.parameters():
            param.requires_grad_(branch_is_trainable)

    # Gates are treated as shared parameters and always remain trainable.
    for gate in unwrapped_model.sidenet.modality_gates.parameters():
        gate.requires_grad_(True)

    # Shared SideNet components remain trainable.
    for param in unwrapped_model.sidenet.fusion.parameters():
        param.requires_grad_(True)
    for param in unwrapped_model.sidenet.injectors.parameters():
        param.requires_grad_(True)
    unwrapped_model.sidenet.fusion_vectors.requires_grad_(True)


def save_base_pi05_checkpoint(model, ckpt_dir: pathlib.Path) -> None:
    safetensors.torch.save_model(model.pi05_model, str(ckpt_dir / "base_pi05.safetensors"))


def save_sidenet_checkpoint(model, ckpt_dir: pathlib.Path) -> None:
    safetensors.torch.save_model(model.sidenet, str(ckpt_dir / "sidenet.safetensors"))


def save_sidenet_branch_checkpoints(model, ckpt_dir: pathlib.Path) -> list[str]:
    branch_dir = ckpt_dir / "sidenet_branches"
    branch_dir.mkdir(parents=True, exist_ok=True)

    saved_branches: list[str] = []
    for branch_name, branch in model.sidenet.branches.items():
        safetensors.torch.save_model(branch, str(branch_dir / f"{branch_name}.safetensors"))
        saved_branches.append(branch_name)

    return saved_branches


def _shared_sidenet_state_dict(model) -> dict[str, torch.Tensor]:
    shared_state: dict[str, torch.Tensor] = {}
    for key, value in model.sidenet.state_dict().items():
        if (
            key == "fusion_vectors"
            or key.startswith("fusion.")
            or key.startswith("injectors.")
            or key.startswith("modality_gates.")
        ):
            shared_state[key] = value.detach().cpu()
    return shared_state


def save_shared_parts_checkpoint(model, ckpt_dir: pathlib.Path) -> None:
    safetensors.torch.save_file(
        _shared_sidenet_state_dict(model),
        str(ckpt_dir / "shared_parts.safetensors"),
    )


def _module_label_from_key(key: str) -> str:
    if key == "fusion_vectors":
        return "fusion_vectors"
    if key.startswith("fusion."):
        return "fusion"
    if key.startswith("injectors."):
        parts = key.split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else "injectors"
    if key.startswith("branches."):
        parts = key.split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else "branches"
    if key.startswith("modality_gates."):
        parts = key.split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else "modality_gates"
    return key.split(".")[0]


def assert_full_sidenet_checkpoint_matches_model(model, path: str, device: torch.device | None = None) -> None:
    checkpoint_state = safetensors.torch.load_file(str(path), device=str(device) if device is not None else "cpu")
    checkpoint_keys = set(checkpoint_state.keys())
    model_keys = set(model.sidenet.state_dict().keys())

    missing_keys = sorted(model_keys - checkpoint_keys)
    unexpected_keys = sorted(checkpoint_keys - model_keys)
    if not missing_keys and not unexpected_keys:
        return

    missing_modules = sorted({_module_label_from_key(key) for key in missing_keys})
    unexpected_modules = sorted({_module_label_from_key(key) for key in unexpected_keys})

    message_lines = [
        "Full SideNet checkpoint does not match the current YAML/model structure.",
    ]
    if missing_modules:
        message_lines.append(f"Missing modules in checkpoint: {missing_modules}")
    if unexpected_modules:
        message_lines.append(f"Unexpected modules in checkpoint: {unexpected_modules}")
    if missing_keys:
        message_lines.append(f"Example missing keys: {missing_keys[:8]}")
    if unexpected_keys:
        message_lines.append(f"Example unexpected keys: {unexpected_keys[:8]}")
    raise AssertionError(" ".join(message_lines))


def load_sidenet_checkpoint(model, path: str, strict: bool = True, device: torch.device | None = None) -> None:
    if strict:
        assert_full_sidenet_checkpoint_matches_model(model, path, device)
    load_kwargs = {"strict": strict}
    if device is not None:
        load_kwargs["device"] = str(device)
    safetensors.torch.load_model(model.sidenet, str(path), **load_kwargs)


def load_sidenet_branch_checkpoints(
    model,
    branch_paths: dict[str, str] | Mapping[str, str],
    strict: bool = True,
    device: torch.device | None = None,
) -> None:
    load_kwargs = {"strict": strict}
    if device is not None:
        load_kwargs["device"] = str(device)

    for branch_name, path in branch_paths.items():
        if branch_name not in model.sidenet.branches:
            raise ValueError(
                f"Unknown SideNet branch `{branch_name}` in branch checkpoint load. "
                f"Available branches: {sorted(model.sidenet.branches.keys())}"
            )
        safetensors.torch.load_model(model.sidenet.branches[branch_name], str(path), **load_kwargs)


def load_shared_parts_checkpoint(model, path: str, device: torch.device) -> None:
    shared_state = safetensors.torch.load_file(str(path), device=str(device))
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
    non_shared_missing = [
        key for key in incompatible.missing_keys if key in model_shared_keys
    ]
    if non_shared_missing:
        raise AssertionError(f"Missing shared checkpoint keys after load: {non_shared_missing}")


def initialize_sidenet_from_config(model, config: _config.TrainConfig, device: torch.device) -> None:
    unwrapped_model = unwrap_model(model)
    ckpt_cfg = config.sidenet.checkpoint

    if ckpt_cfg.load_sidenet_full_path is not None:
        load_sidenet_checkpoint(
            unwrapped_model,
            ckpt_cfg.load_sidenet_full_path,
            strict=ckpt_cfg.strict_sidenet_load,
            device=device,
        )

    if ckpt_cfg.load_sidenet_branch_paths:
        load_sidenet_branch_checkpoints(
            unwrapped_model,
            ckpt_cfg.load_sidenet_branch_paths,
            strict=ckpt_cfg.strict_sidenet_load,
            device=device,
        )

    if ckpt_cfg.load_shared_parts_separately:
        if ckpt_cfg.shared_parts_path is None:
            raise ValueError(
                "`config.sidenet.checkpoint.shared_parts_path` must be set when "
                "`load_shared_parts_separately=True`."
            )
        load_shared_parts_checkpoint(unwrapped_model, ckpt_cfg.shared_parts_path, device)


# FIX: checkpoint logging now writes checkpoint step to TensorBoard instead of wandb.
def save_checkpoint(model, optimizer, global_step, config, is_main, data_config, writer: SummaryWriter | None = None):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        checkpoint_dir = get_training_checkpoint_dir(config)
        final_ckpt_dir = checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        model_to_save = unwrap_model(model)
        ckpt_cfg = config.sidenet.checkpoint

        saved_branches: list[str] = []
        if ckpt_cfg.save_base_pi05:
            save_base_pi05_checkpoint(model_to_save, tmp_ckpt_dir)
        if ckpt_cfg.save_sidenet_full:
            save_sidenet_checkpoint(model_to_save, tmp_ckpt_dir)
        if ckpt_cfg.save_sidenet_branches:
            saved_branches = save_sidenet_branch_checkpoints(model_to_save, tmp_ckpt_dir)
        if ckpt_cfg.save_shared_parts:
            save_shared_parts_checkpoint(model_to_save, tmp_ckpt_dir)

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
            "saved_parts": {
                "base_pi05": ckpt_cfg.save_base_pi05,
                "sidenet_full": ckpt_cfg.save_sidenet_full,
                "sidenet_branches": saved_branches,
                "shared_parts": ckpt_cfg.save_shared_parts,
            },
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to TensorBoard
        if writer is not None:
            writer.add_scalar("checkpoint/step", global_step, global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        model_to_load = unwrap_model(model)
        loaded_any_model_weights = False

        logging.info("Loading SideNet/base-pi05 checkpoint parts...")

        base_pi05_path = ckpt_dir / "base_pi05.safetensors"
        if base_pi05_path.exists():
            safetensors.torch.load_model(model_to_load.pi05_model, str(base_pi05_path), device=str(device))
            loaded_any_model_weights = True
            logging.info("Loaded base pi05 weights from split checkpoint")

        sidenet_path = ckpt_dir / "sidenet.safetensors"
        if sidenet_path.exists():
            safetensors.torch.load_model(model_to_load.sidenet, str(sidenet_path), device=str(device))
            loaded_any_model_weights = True
            logging.info("Loaded full SideNet weights from split checkpoint")

        branch_dir = ckpt_dir / "sidenet_branches"
        if branch_dir.exists():
            branch_paths = {
                path.stem: str(path)
                for path in branch_dir.glob("*.safetensors")
            }
            if branch_paths:
                load_sidenet_branch_checkpoints(model_to_load, branch_paths, strict=True, device=device)
                loaded_any_model_weights = True
                logging.info("Loaded %d SideNet branch checkpoints", len(branch_paths))

        shared_parts_path = ckpt_dir / "shared_parts.safetensors"
        if shared_parts_path.exists():
            load_shared_parts_checkpoint(model_to_load, str(shared_parts_path), device)
            loaded_any_model_weights = True
            logging.info("Loaded standalone shared SideNet parts from split checkpoint")

        if not loaded_any_model_weights:
            raise FileNotFoundError(f"No split model checkpoint parts found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)
    checkpoint_dir = get_training_checkpoint_dir(config)

    # FIX: initialize checkpoint directory and TensorBoard writer for the SideNet training path.
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {checkpoint_dir}")

    # FIX: initialize TensorBoard writer (only on main process).
    writer = None
    if is_main:
        writer = init_tensorboard(config, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)

    # FIX: preview sample images through TensorBoard while staying on the
    # standard `(Observation, actions)` loader contract.
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_observation, sample_actions = next(iter(sample_data_loader))
        sample_batch = sample_observation.to_dict()
        sample_batch["actions"] = sample_actions

        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item and
            # write them as HWC images to TensorBoard.
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu()
            if writer is not None:
                writer.add_image(f"camera_views/sample_{i}", img_concatenated, 0, dataformats="HWC")

        # Clear sample batch from memory aggressively
        del sample_batch, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # FIX: this training entrypoint is dedicated to PI05withSideNet and reads all
    # model selection parameters from `config.sidenet`.
    if not config.sidenet.enabled:
        raise ValueError(
            "`config.sidenet.enabled` must be True when using `sidenet/train.py`."
        )
    if config.pytorch_weight_path is None:
        raise ValueError(
            "`config.pytorch_weight_path` must point to a pretrained pi05 checkpoint "
            "when training the SideNet wrapper."
        )

    # FIX: read the SideNet yaml path from training config instead of hardcoding it.
    sidenet_config_path = (
        pathlib.Path(config.sidenet.config_path).expanduser().resolve()
        if config.sidenet.config_path is not None
        else pathlib.Path(__file__).with_name("sidenet_config.yaml")
    )
    # FIX: instantiate PI05withSideNet instead of the base PI0Pytorch model.
    model = PI05withSideNet(
        sidenet_config_path=str(sidenet_config_path),
        pi05_config_name=config.name,
        pi05_weights_path=os.path.join(config.pytorch_weight_path, "model.safetensors"),
    ).to(device)

    # FIX: optionally initialize SideNet from its own checkpoints before DDP wrapping.
    if not resuming:
        initialize_sidenet_from_config(model, config, device)

    # FIX: freeze the base pi05 model and/or selected SideNet branches before
    # optimizer creation so only the requested parameters are trained.
    configure_trainable_parameters(model, config)

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

    # FIX: base pi05 weights are loaded inside PI05withSideNet construction.
    logging.info(f"Loaded base pi05 weights from {config.pytorch_weight_path}")

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters remain after applying SideNet training configuration.")

    optim = torch.optim.AdamW(
        trainable_parameters,
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {config.pytorch_training_precision}")
        # FIX: log SideNet-specific training knobs for experiment traceability.
        logging.info(
            "SideNet config: path=%s train_pi05=%s modalities=%s trainable_branches=%s backbone_injector=%s expert_injector=%s",
            sidenet_config_path,
            config.sidenet.train_pi05,
            tuple(config.sidenet.modalities),
            tuple(config.sidenet.trainable_branches),
            config.sidenet.use_backbone_injector,
            config.sidenet.use_expert_injector,
        )
        logging.info(
            "SideNet checkpoint config: save_base_pi05=%s save_sidenet_full=%s save_sidenet_branches=%s "
            "save_shared_parts=%s load_sidenet_full=%s load_branch_count=%d load_shared_parts_separately=%s "
            "checkpoint_dir=%s",
            config.sidenet.checkpoint.save_base_pi05,
            config.sidenet.checkpoint.save_sidenet_full,
            config.sidenet.checkpoint.save_sidenet_branches,
            config.sidenet.checkpoint.save_shared_parts,
            config.sidenet.checkpoint.load_sidenet_full_path is not None,
            len(config.sidenet.checkpoint.load_sidenet_branch_paths),
            config.sidenet.checkpoint.load_shared_parts_separately,
            checkpoint_dir,
        )

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        # FIX: this remains a known gap; `loader.set_epoch` is not formally exposed
        # through the current custom loader stack yet.
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        # FIX: iterate over the standard `(Observation, actions)` loader output.
        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # FIX: keep only the configured SideNet modalities inside
            # `Observation.modalities`, and route the standard Observation object
            # directly into the wrapper model.
            observation = select_observation_modalities(
                observation,
                modalities=tuple(config.sidenet.modalities),
            )
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Forward pass
            # FIX: the wrapper model now reads SideNet inputs from
            # `observation.modalities`, so we stay on the standard model
            # interface and only pass SideNet training switches here.
            losses = model(
                observation,
                actions,
                train_pi05=config.sidenet.train_pi05,
                use_backbone_injector=config.sidenet.use_backbone_injector,
                use_expert_injector=config.sidenet.use_expert_injector,
            )
            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            # Backward pass
            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # FIX: log training scalars to TensorBoard instead of wandb.
                if writer is not None and len(infos) > 0:
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar("train/learning_rate", avg_lr, global_step)
                    writer.add_scalar("train/time_per_step", elapsed / config.log_interval, global_step)
                    if avg_grad_norm is not None:
                        writer.add_scalar("train/grad_norm", avg_grad_norm, global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config, writer=writer)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    if writer is not None:
        writer.close()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
