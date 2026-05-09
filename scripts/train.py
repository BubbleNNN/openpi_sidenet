import dataclasses
import copy
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb
import time

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.checker as _checker
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms
import sidenet.jax_impl.checkpointing as sidenet_jax_checkpointing
try:
    from tensorboardX import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    _TENSORBOARD_AVAILABLE = False
if _TENSORBOARD_AVAILABLE:
    
    class TensorboardLogger:
        def __init__(self, config):
            self.writer = SummaryWriter(log_dir=str(config.checkpoint_dir / "tensorboard"))
            
        def log(self, step:int, scalars: dict[str, float]):
            for key, value in scalars.items():
                self.writer.add_scalar(key, value, step)

        def log_text(self, step: int, tag: str, text: str):
            self.writer.add_text(tag, text, step)

        def log_image(self, step: int, tag: str, image: np.ndarray):
            self.writer.add_image(tag, image, step, dataformats="HWC")
        
        def close(self):
            self.writer.close()
else:
    class TensorboardLogger:
        def __init__(self, config):
           self.writer = None
        
        def log(self, step:int, scalars: dict[str, float]):
            pass

        def log_text(self, step: int, tag: str, text: str):
            pass

        def log_image(self, step: int, tag: str, image: np.ndarray):
            pass
        
        def close(self):
            pass


def _shape_numel(shape: tuple[int, ...]) -> int:
    if not shape:
        return 1
    return int(np.prod(shape))


def _leaf_array(value):
    if hasattr(value, "value"):
        value = value.value
    if value is None or not hasattr(value, "shape"):
        return None
    return value


def _flatten_arrays(tree: Any) -> dict[str, Any]:
    if tree is None:
        return {}
    if hasattr(tree, "to_pure_dict"):
        tree = tree.to_pure_dict()
    flat = traverse_util.flatten_dict(tree, sep="/")
    return {
        key: array
        for key, value in flat.items()
        if (array := _leaf_array(value)) is not None
    }


def _component_from_param_path(path: str) -> str:
    parts = path.split("/")
    if not parts:
        return "unknown"
    if "injection_gate" in path:
        return "injection_gate"
    if parts[0] == "sidenet":
        if len(parts) >= 3 and parts[1] in {"tokenizers", "branches"}:
            return "/".join(parts[:3])
        if len(parts) >= 2:
            return "/".join(parts[:2])
        return "sidenet"
    if "norm" in path.lower():
        if parts[:2] == ["PaliGemma", "llm"] and len(parts) >= 3:
            return "/".join(["norm", *parts[:3]])
        if len(parts) >= 2:
            return "/".join(["norm", *parts[:2]])
        return "norm"
    return "/".join(parts[: min(3, len(parts))])


def _component_l2_norms(tree: Any, *, prefix: str) -> dict[str, Any]:
    grouped: dict[str, Any] = {}
    for path, value in _flatten_arrays(tree).items():
        component = _component_from_param_path(path)
        value_f32 = jnp.asarray(value, dtype=jnp.float32)
        sq_norm = jnp.sum(jnp.square(value_f32))
        grouped[component] = grouped.get(component, jnp.asarray(0.0, dtype=jnp.float32)) + sq_norm
    return {f"{prefix}/{component}": jnp.sqrt(sq_norm) for component, sq_norm in sorted(grouped.items())}


def _param_entries(tree: Any) -> list[dict[str, Any]]:
    entries = []
    for path, value in sorted(_flatten_arrays(tree).items()):
        shape = tuple(int(dim) for dim in value.shape)
        entries.append(
            {
                "path": path,
                "shape": shape,
                "dtype": str(getattr(value, "dtype", "unknown")),
                "numel": _shape_numel(shape),
                "component": _component_from_param_path(path),
            }
        )
    return entries


def _summarize_entries(entries: list[dict[str, Any]]) -> tuple[int, int, dict[str, int]]:
    total_numel = sum(int(entry["numel"]) for entry in entries)
    by_component: dict[str, int] = {}
    for entry in entries:
        by_component[entry["component"]] = by_component.get(entry["component"], 0) + int(entry["numel"])
    return len(entries), total_numel, dict(sorted(by_component.items()))


def _format_entries_table(title: str, entries: list[dict[str, Any]], *, max_rows: int | None = None) -> str:
    shown = entries if max_rows is None else entries[:max_rows]
    lines = [title, "path\tshape\tdtype\tnumel\tcomponent"]
    for entry in shown:
        lines.append(
            f"{entry['path']}\t{entry['shape']}\t{entry['dtype']}\t{entry['numel']}\t{entry['component']}"
        )
    if max_rows is not None and len(entries) > max_rows:
        lines.append(f"... {len(entries) - max_rows} more")
    return "\n".join(lines)


def _write_diagnostics_text(config: _config.TrainConfig, filename: str, text: str):
    diagnostics_dir = config.checkpoint_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    (diagnostics_dir / filename).write_text(text)


def _initialization_report(reference_params: at.Params, loaded_params: at.Params, *, resumed: bool) -> dict[str, Any]:
    if resumed:
        return {"resumed": True, "loaded": [], "random_initialized": []}
    flat_ref = _flatten_arrays(reference_params)
    flat_loaded = _flatten_arrays(loaded_params)
    loaded_keys = set(flat_loaded)
    loaded_entries = []
    random_entries = []
    for path, value in sorted(flat_ref.items()):
        shape = tuple(int(dim) for dim in value.shape)
        entry = {
            "path": path,
            "shape": shape,
            "dtype": str(getattr(value, "dtype", "unknown")),
            "numel": _shape_numel(shape),
            "component": _component_from_param_path(path),
        }
        if path in loaded_keys:
            loaded_entries.append(entry)
        else:
            random_entries.append(entry)
    return {
        "resumed": False,
        "loaded": loaded_entries,
        "random_initialized": random_entries,
    }


def _log_initialization_report(tb_logger: TensorboardLogger, config: _config.TrainConfig, report: dict[str, Any]):
    if report.get("resumed"):
        text = "Training was resumed from an existing checkpoint; base weight initialization was skipped."
        tb_logger.log_text(0, "init/status", text)
        _write_diagnostics_text(config, "initialization_report.txt", text)
        return

    loaded = report["loaded"]
    random_initialized = report["random_initialized"]
    loaded_count, loaded_numel, loaded_by_component = _summarize_entries(loaded)
    random_count, random_numel, random_by_component = _summarize_entries(random_initialized)

    scalars = {
        "init/loaded_num_leaves": loaded_count,
        "init/loaded_numel": loaded_numel,
        "init/random_initialized_num_leaves": random_count,
        "init/random_initialized_numel": random_numel,
    }
    scalars.update({f"init/loaded_numel_by_component/{key}": value for key, value in loaded_by_component.items()})
    scalars.update(
        {f"init/random_initialized_numel_by_component/{key}": value for key, value in random_by_component.items()}
    )
    tb_logger.log(0, scalars)

    full_text = "\n\n".join(
        [
            f"Loaded from checkpoint: leaves={loaded_count}, numel={loaded_numel}",
            _format_entries_table("Loaded from checkpoint", loaded),
            f"Random / freshly initialized: leaves={random_count}, numel={random_numel}",
            _format_entries_table("Random / freshly initialized", random_initialized),
        ]
    )
    tb_logger.log_text(
        0,
        "init/params_loaded_vs_initialized",
        "\n\n".join(
            [
                f"Loaded from checkpoint: leaves={loaded_count}, numel={loaded_numel}",
                _format_entries_table("Loaded preview", loaded, max_rows=200),
                f"Random / freshly initialized: leaves={random_count}, numel={random_numel}",
                _format_entries_table("Random preview", random_initialized, max_rows=200),
            ]
        ),
    )
    _write_diagnostics_text(config, "initialization_report.txt", full_text)


def _log_trainable_params(tb_logger: TensorboardLogger, config: _config.TrainConfig, params: nnx.State):
    trainable_params = params.filter(config.trainable_filter)
    entries = _param_entries(trainable_params)
    count, numel, by_component = _summarize_entries(entries)
    scalars = {
        "params/trainable_num_leaves": count,
        "params/trainable_numel": numel,
    }
    scalars.update({f"params/trainable_numel_by_component/{key}": value for key, value in by_component.items()})
    tb_logger.log(0, scalars)

    full_text = "\n\n".join(
        [
            f"Trainable params: leaves={count}, numel={numel}",
            _format_entries_table("Trainable parameters", entries),
        ]
    )
    tb_logger.log_text(
        0,
        "params/trainable",
        "\n\n".join(
            [
                f"Trainable params: leaves={count}, numel={numel}",
                _format_entries_table("Trainable preview", entries, max_rows=250),
            ]
        ),
    )
    _write_diagnostics_text(config, "trainable_params.txt", full_text)


def _as_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        value = jax.device_get(value)
    except Exception:
        pass
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        return np.asarray(value)
    except Exception:
        return None


def _flatten_data_tree(tree: Any, prefix: str = "") -> dict[str, Any]:
    if tree is None:
        return {}
    if hasattr(tree, "to_dict"):
        tree = tree.to_dict()
    if isinstance(tree, dict):
        result = {}
        for key, value in tree.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            result.update(_flatten_data_tree(value, child_prefix))
        return result
    return {prefix: tree}


def _safe_tb_tag(text: str) -> str:
    return text.replace(" ", "_").replace(":", "_").replace("[", "_").replace("]", "_")


def _prepare_image_for_tb(array: np.ndarray) -> np.ndarray | None:
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3:
        return None
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] not in (1, 3, 4):
        return None
    if array.dtype == np.uint8:
        return array
    image = array.astype(np.float32)
    finite = np.isfinite(image)
    if not np.any(finite):
        return None
    min_value = float(np.min(image[finite]))
    max_value = float(np.max(image[finite]))
    if min_value < 0.0 or max_value > 1.0:
        denom = max(max_value - min_value, 1e-6)
        image = (image - min_value) / denom
    return np.clip(image, 0.0, 1.0)


def _log_vector_dim_stats(
    tb_logger: TensorboardLogger,
    step: int,
    tag_prefix: str,
    array: np.ndarray,
    *,
    max_dims: int = 64,
):
    if array.size == 0 or not np.issubdtype(array.dtype, np.number):
        return
    array = array.astype(np.float32)
    if array.ndim == 0:
        tb_logger.log(
            step,
            {
                f"{tag_prefix}/mean": float(np.mean(array)),
                f"{tag_prefix}/std": float(np.std(array)),
            },
        )
        return
    flat = array.reshape(-1, array.shape[-1])
    means = np.nanmean(flat, axis=0)
    stds = np.nanstd(flat, axis=0)
    scalars = {}
    for dim in range(min(flat.shape[-1], max_dims)):
        scalars[f"{tag_prefix}/dim_{dim:02d}/mean"] = float(means[dim])
        scalars[f"{tag_prefix}/dim_{dim:02d}/std"] = float(stds[dim])
    tb_logger.log(step, scalars)


def _log_data_tree_snapshot(
    tb_logger: TensorboardLogger,
    step: int,
    tree: Any,
    *,
    tag_prefix: str,
    max_images: int = 8,
    log_shapes: bool = True,
):
    image_count = 0
    shape_lines = []
    for path, value in sorted(_flatten_data_tree(tree).items()):
        array = _as_numpy(value)
        if array is None:
            continue
        shape_lines.append(f"{path}: shape={array.shape}, dtype={array.dtype}")
        lower_path = path.lower()
        if "image" in lower_path and image_count < max_images:
            image = _prepare_image_for_tb(array)
            if image is not None:
                tb_logger.log_image(step, f"{tag_prefix}/images/{_safe_tb_tag(path)}", image)
                image_count += 1
        if any(name in lower_path for name in ("state", "action", "actions", "ft_sensor", "force_torque")):
            if "image" not in lower_path and np.issubdtype(array.dtype, np.number):
                _log_vector_dim_stats(
                    tb_logger,
                    step,
                    f"{tag_prefix}/stats/{_safe_tb_tag(path)}",
                    array,
                )
    if log_shapes and shape_lines:
        tb_logger.log_text(step, f"{tag_prefix}/shapes", "\n".join(shape_lines[:300]))


def _copy_tree(tree: Any) -> Any:
    try:
        return copy.deepcopy(tree)
    except Exception:
        return jax.tree.map(
            lambda x: np.array(x).copy() if hasattr(x, "shape") else x,
            tree,
            is_leaf=lambda x: x is None,
        )


def _apply_transforms(data: dict[str, Any], transforms: list[Any]) -> dict[str, Any]:
    out = _copy_tree(data)
    for transform in transforms:
        out = transform(out)
    return out


def _log_transform_pipeline_snapshot(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    tb_logger: TensorboardLogger,
):
    if data_config.rlds_data_dir is not None:
        tb_logger.log_text(0, "data/raw_pipeline/status", "RLDS raw pipeline snapshot is not implemented.")
        return
    try:
        raw_dataset = _data_loader.create_torch_dataset(
            data_config,
            action_horizon=config.model.action_horizon,
            model_config=config.model,
        )
        raw_sample = raw_dataset[0]
        _log_data_tree_snapshot(tb_logger, 0, raw_sample, tag_prefix="data/raw")

        repacked = _apply_transforms(raw_sample, list(data_config.repack_transforms.inputs))
        _log_data_tree_snapshot(tb_logger, 0, repacked, tag_prefix="data/after_repack")

        robot_inputs = _apply_transforms(repacked, list(data_config.data_transforms.inputs))
        _log_data_tree_snapshot(tb_logger, 0, robot_inputs, tag_prefix="data/after_robot_transforms")

        norm_stats = {} if data_config.norm_stats is None else data_config.norm_stats
        normalized = _apply_transforms(
            robot_inputs,
            [_transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)],
        )
        _log_data_tree_snapshot(tb_logger, 0, normalized, tag_prefix="data/after_normalize")

        model_inputs = _apply_transforms(normalized, list(data_config.model_transforms.inputs))
        _log_data_tree_snapshot(tb_logger, 0, model_inputs, tag_prefix="data/after_model_transforms")
    except Exception as exc:
        logging.exception("Failed to log data transform pipeline snapshot")
        tb_logger.log_text(0, "data/raw_pipeline/error", repr(exc))


def _log_model_input_batch_snapshot(
    tb_logger: TensorboardLogger,
    step: int,
    batch: tuple[_model.Observation, _model.Actions],
    *,
    tag_prefix: str = "data/model_input",
    log_images: bool = False,
    log_shapes: bool = False,
):
    observation, actions = batch
    payload = observation.to_dict()
    payload["actions"] = actions
    _log_data_tree_snapshot(
        tb_logger,
        step,
        payload,
        tag_prefix=tag_prefix,
        max_images=8 if log_images else 0,
        log_shapes=log_shapes,
    )
    token_mask = _as_numpy(observation.tokenized_prompt_mask)
    if token_mask is not None:
        tb_logger.log(
            step,
            {
                f"{tag_prefix}/tokenized_prompt/valid_tokens_mean": float(np.mean(np.sum(token_mask, axis=-1))),
                f"{tag_prefix}/tokenized_prompt/valid_tokens_std": float(np.std(np.sum(token_mask, axis=-1))),
            },
        )


def init_logging():
    """Custom logging format for better readability."""
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
    logger.handlers[0].setFormatter(formatter)

class LogFile:
    """text file logging for loss"""
    
    def __init__(self, config):
        ckpt_dir = config.checkpoint_dir
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
        self.log_file = ckpt_dir / "loss.log"
        
    def write(self, step: int, message: str):
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(self.log_file, "a") as f:
            f.write(f"{time_str} Step {step}: {message}\n")

def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(
    loader: _weight_loaders.WeightLoader, params_shape: at.Params, *, action_dim: int | None = None
) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any, dict[str, Any]]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding, _initialization_report({}, {}, resumed=True)

    reference_params = train_state_shape.params.to_pure_dict()
    partial_params = _load_weights_and_validate(config.weight_loader, reference_params, action_dim = config.model.action_dim)
    init_report = _initialization_report(reference_params, partial_params, resumed=False)
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding, init_report


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    info.update(_component_l2_norms(grads, prefix="grad_norm/component"))
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    log_file = LogFile(config)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    extra_asset_callbacks = []
    if (callback := sidenet_jax_checkpointing.maybe_get_asset_callback(config.model)) is not None:
        extra_asset_callbacks.append(callback)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    train_checker = None
    if config.training_checker.enabled:
        checker_dir = config.checkpoint_dir / config.training_checker.output_subdir
        train_checker = _checker.TrainingChecker(config.training_checker, checker_dir)
        logging.info("Training checker enabled; reports will be written to %s", checker_dir)

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding, init_report = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    tb_logger = TensorboardLogger(config)
    _log_initialization_report(tb_logger, config, init_report)
    _log_trainable_params(tb_logger, config, train_state.params)
    _log_transform_pipeline_snapshot(config, data_loader.data_config(), tb_logger)
    _log_model_input_batch_snapshot(
        tb_logger,
        0,
        batch,
        tag_prefix="data/model_input_initial",
        log_images=True,
        log_shapes=True,
    )

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )
    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            display_info = {
                key: value for key, value in reduced_info.items()
                if not key.startswith("grad_norm/component/")
            }
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in display_info.items())
            pbar.write(f"Step {step}: {info_str}")
            log_file.write(step, info_str)
            wandb.log(display_info, step=step)
            tb_logger.log(step, reduced_info)
            _log_model_input_batch_snapshot(tb_logger, step, batch)
            infos = []

        if train_checker is not None and train_checker.should_run(step):
            try:
                with sharding.set_mesh(mesh):
                    checker_report = train_checker.run(config, train_rng, train_state, batch)
                train_checker.write(step, checker_report)
                checker_scalars = train_checker.summarize_scalars(checker_report)
                if checker_scalars:
                    wandb.log(checker_scalars, step=step)
                    tb_logger.log(step, checker_scalars)
            except Exception:
                logging.exception("Training checker failed at step %d", step)
        batch = next(data_iter)
        # # 添加以下代码来查看第一个batch的统计信息
        # observation, actions = batch
        # actions_arr = np.asarray(actions)
        # actions_flat = actions_arr.reshape(-1, actions_arr.shape[-1])
        # print(f"\n{'='*60}")
        # print(f"[Initial Batch ] Action Per-joint statistics")
        # print(f"Shape: batch={actions_arr.shape[0]}, horizon = {actions_arr.shape[1]}, joints={actions_arr.shape[2]}")
        # print(f"{'='*60}")
        # print(f)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(
                checkpoint_manager,
                train_state,
                data_loader,
                step,
                extra_asset_callbacks=extra_asset_callbacks,
            )

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    tb_logger.close()


if __name__ == "__main__":
    main(_config.cli())
