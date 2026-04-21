from __future__ import annotations

import dataclasses
import functools
import json
import logging
import pathlib
from typing import Any

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import model as _model
from openpi.models import pi0 as _pi0

try:
    from sidenet.jax import pi05_with_sidenet as _pi05_with_sidenet
except Exception:  # pragma: no cover - checker should still work in clean openpi.
    _pi05_with_sidenet = None

logger = logging.getLogger(__name__)

_PI0_COMPONENT_NAMES = (
    "prefix_tokens",
    "suffix_tokens",
    "prefix_out",
    "suffix_out",
    "v_t",
)

_PI05_WITH_SIDENET_COMPONENT_NAMES = (
    "prefix_tokens",
    "prefix_out",
    "text_features",
    "ft_sensor",
    "suffix_tokens_pre_side",
    "side_tokens",
    "suffix_tokens_post_side",
    "suffix_out",
    "v_t",
    "sidenet/branches/ft_sensor",
    "sidenet/concat_tokens",
    "sidenet/fused_tokens",
    "sidenet/text_cross_attn",
    "sidenet/output_mlp",
)


@dataclasses.dataclass(frozen=True)
class TrainingCheckerConfig:
    enabled: bool = False
    interval: int = 100
    initial_consecutive_steps: int = 5
    output_subdir: str = "training_checker"
    max_per_sample: int = 8
    max_leaf_paths: int = 32
    max_component_paths: int = 16
    compute_component_ablation: bool = True
    compute_component_scale_gradients: bool = True
    compute_per_sample_gradients: bool = True
    log_full_per_sample_losses: bool = True


@dataclasses.dataclass(frozen=True)
class DebugForwardResult:
    chunked_loss: jax.Array
    debug_tensors: dict[str, jax.Array]
    component_names: tuple[str, ...]


def _reduce_per_sample_loss(loss: jax.Array) -> jax.Array:
    if loss.ndim <= 1:
        return loss
    axes = tuple(range(1, loss.ndim))
    return jnp.mean(loss, axis=axes)


def _to_float(x: Any) -> float:
    return float(np.asarray(jax.device_get(x)))


def _to_python(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "shape") or isinstance(value, np.ndarray):
        arr = np.asarray(jax.device_get(value))
        if arr.ndim == 0:
            return arr.item()
        return arr.tolist()
    return value


def _tree_to_flat_arrays(tree: Any) -> dict[str, jax.Array]:
    if tree is None:
        return {}
    if hasattr(tree, "to_pure_dict"):
        tree = tree.to_pure_dict()
    flat = traverse_util.flatten_dict(tree, sep="/")
    return {path: jnp.asarray(value) for path, value in flat.items() if hasattr(value, "shape")}


def _array_l2_norm(x: jax.Array) -> jax.Array:
    x = jnp.asarray(x, dtype=jnp.float32)
    return jnp.sqrt(jnp.sum(jnp.square(x)))


def _tensor_stats(x: jax.Array) -> dict[str, Any]:
    x = jnp.asarray(x)
    x_f32 = x.astype(jnp.float32)
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "l2_norm": _to_float(_array_l2_norm(x_f32)),
        "mean": _to_float(jnp.mean(x_f32)),
        "std": _to_float(jnp.std(x_f32)),
        "abs_max": _to_float(jnp.max(jnp.abs(x_f32))),
        "min": _to_float(jnp.min(x_f32)),
        "max": _to_float(jnp.max(x_f32)),
        "nan_count": int(np.asarray(jax.device_get(jnp.sum(jnp.isnan(x_f32))))),
        "inf_count": int(np.asarray(jax.device_get(jnp.sum(jnp.isinf(x_f32))))),
    }


def _group_path(path: str) -> str:
    parts = path.split("/")
    if not parts:
        return path

    if parts[0] == "sidenet":
        if len(parts) >= 3 and parts[1] == "branches":
            return "/".join(parts[:3])
        if len(parts) >= 2:
            return "/".join(parts[:2])
        return parts[0]

    if parts[0] == "base_model":
        if len(parts) >= 3 and parts[1] == "PaliGemma":
            return "/".join(parts[:3])
        if len(parts) >= 2:
            return "/".join(parts[:2])
        return parts[0]

    return "/".join(parts[: min(3, len(parts))])


def _aggregate_norms(flat_arrays: dict[str, jax.Array]) -> dict[str, float]:
    grouped: dict[str, jax.Array] = {}
    for path, value in flat_arrays.items():
        key = _group_path(path)
        sq_norm = jnp.sum(jnp.square(jnp.asarray(value, dtype=jnp.float32)))
        grouped[key] = grouped.get(key, jnp.asarray(0.0, dtype=jnp.float32)) + sq_norm
    return {key: _to_float(jnp.sqrt(value)) for key, value in sorted(grouped.items())}


def _topk_leaf_norms(flat_arrays: dict[str, jax.Array], k: int) -> dict[str, float]:
    scored = [(path, _to_float(_array_l2_norm(value))) for path, value in flat_arrays.items()]
    scored.sort(key=lambda item: item[1], reverse=True)
    return {path: norm for path, norm in scored[:k]}


def _slice_batch(tree: Any, index: int) -> Any:
    return jax.tree.map(
        lambda x: x[index : index + 1] if x is not None and hasattr(x, "shape") else x,
        tree,
        is_leaf=lambda x: x is None,
    )


def _tap_debug_tensor(
    debug_tensors: dict[str, jax.Array] | None,
    component_scales: dict[str, jax.Array] | None,
    name: str,
    x: jax.Array,
) -> jax.Array:
    if debug_tensors is not None:
        debug_tensors[name] = x
    if component_scales is not None and name in component_scales:
        scale = jnp.asarray(component_scales[name], dtype=x.dtype)
        x = x * scale
    return x


def _debug_sidenet_call(
    module: Any,
    modality_inputs: dict[str, jnp.ndarray],
    text_embeddings: jnp.ndarray,
    text_mask: jnp.ndarray | None = None,
    deterministic: bool = True,
    *,
    debug_tensors: dict[str, jax.Array] | None = None,
    component_scales: dict[str, jax.Array] | None = None,
) -> tuple[jnp.ndarray, dict[str, jax.Array]]:
    side_debug_tensors = {} if debug_tensors is None else debug_tensors
    branch_outputs = []
    for name, _ in sorted(module.branch_input_dims):
        if name not in modality_inputs:
            raise KeyError(
                f"Missing modality input for branch `{name}`. Available inputs: {sorted(modality_inputs.keys())}"
            )

        branch_input = modality_inputs[name]
        if name == "ft_sensor" and branch_input.ndim == 3 and branch_input.shape[1] != 1:
            raise ValueError(
                "JAX SideNet currently expects single-frame `ft_sensor` input "
                "with shape `(B, 12)` or `(B, 1, 12)`, not a temporal window."
            )

        out = module.branches[name](branch_input, deterministic=deterministic)
        out = _tap_debug_tensor(side_debug_tensors, component_scales, f"branches/{name}", out)
        branch_outputs.append(out)

    concat_tokens = jnp.concatenate(branch_outputs, axis=1)
    concat_tokens = _tap_debug_tensor(side_debug_tensors, component_scales, "concat_tokens", concat_tokens)
    fused_tokens = module.concat_self_attn(concat_tokens, deterministic=deterministic)
    fused_tokens = _tap_debug_tensor(side_debug_tensors, component_scales, "fused_tokens", fused_tokens)

    output = module.text_cross_attn(fused_tokens, text_embeddings, text_mask, deterministic=deterministic)
    output = _tap_debug_tensor(side_debug_tensors, component_scales, "text_cross_attn", output)

    output = module.output_mlp(output)
    output = _tap_debug_tensor(side_debug_tensors, component_scales, "output_mlp", output)
    return output, side_debug_tensors


def _pi0_debug_forward(
    model: _pi0.Pi0,
    rng: jax.Array,
    observation: _model.Observation,
    actions: _model.Actions,
    *,
    train: bool,
    component_scales: dict[str, jax.Array] | None = None,
) -> DebugForwardResult:
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

    debug_tensors: dict[str, jax.Array] = {}

    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    debug_tensors["x_t"] = x_t
    debug_tensors["u_t"] = u_t
    debug_tensors["time"] = time

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_tokens = _tap_debug_tensor(debug_tensors, component_scales, "prefix_tokens", prefix_tokens)

    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, time)
    suffix_tokens = _tap_debug_tensor(debug_tensors, component_scales, "suffix_tokens", suffix_tokens)

    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens],
        mask=attn_mask,
        positions=positions,
        adarms_cond=[None, adarms_cond],
    )
    prefix_out = _tap_debug_tensor(debug_tensors, component_scales, "prefix_out", prefix_out)
    suffix_out = _tap_debug_tensor(debug_tensors, component_scales, "suffix_out", suffix_out)

    v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
    v_t = _tap_debug_tensor(debug_tensors, component_scales, "v_t", v_t)
    chunked_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
    return DebugForwardResult(chunked_loss, debug_tensors, _PI0_COMPONENT_NAMES)


def _pi05_with_sidenet_debug_forward(
    model: Any,
    rng: jax.Array,
    observation: _model.Observation,
    actions: _model.Actions,
    *,
    train: bool,
    component_scales: dict[str, jax.Array] | None = None,
) -> DebugForwardResult:
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

    debug_tensors: dict[str, jax.Array] = {}
    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    debug_tensors["x_t"] = x_t
    debug_tensors["u_t"] = u_t
    debug_tensors["time"] = time

    prefix_tokens, prefix_mask, prefix_ar_mask = model.base_model.embed_prefix(observation)
    prefix_tokens = _tap_debug_tensor(debug_tensors, component_scales, "prefix_tokens", prefix_tokens)
    prefix_out, _ = model._run_prefix_vlm(
        prefix_tokens,
        prefix_mask,
        prefix_ar_mask,
        deterministic=not train,
    )
    prefix_out = _tap_debug_tensor(debug_tensors, component_scales, "prefix_out", prefix_out)

    text_features, text_mask = model._extract_text_features(prefix_out, observation)
    text_features = _tap_debug_tensor(debug_tensors, component_scales, "text_features", text_features)
    ft_sensor = model._resolve_ft_sensor(observation)
    ft_sensor = _tap_debug_tensor(debug_tensors, component_scales, "ft_sensor", ft_sensor)

    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.base_model.embed_suffix(observation, x_t, time)
    suffix_tokens = _tap_debug_tensor(debug_tensors, component_scales, "suffix_tokens_pre_side", suffix_tokens)

    side_component_scales = None
    if component_scales is not None:
        side_component_scales = {
            key.removeprefix("sidenet/"): value
            for key, value in component_scales.items()
            if key.startswith("sidenet/")
        } or None

    side_tokens, side_debug = model.sidenet(
        {"ft_sensor": ft_sensor},
        text_features,
        text_mask,
        deterministic=not train,
        method=functools.partial(
            _debug_sidenet_call,
            debug_tensors={},
            component_scales=side_component_scales,
        ),
    )
    for key, value in side_debug.items():
        debug_tensors[f"sidenet/{key}"] = value
    side_tokens = _tap_debug_tensor(debug_tensors, component_scales, "side_tokens", side_tokens)

    suffix_tokens, suffix_mask, suffix_ar_mask = model._prepend_side_tokens(
        side_tokens,
        suffix_tokens,
        suffix_mask,
        suffix_ar_mask,
    )
    suffix_tokens = _tap_debug_tensor(debug_tensors, component_scales, "suffix_tokens_post_side", suffix_tokens)

    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = model.base_model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens],
        mask=attn_mask,
        positions=positions,
        adarms_cond=[None, adarms_cond],
    )
    suffix_out = _tap_debug_tensor(debug_tensors, component_scales, "suffix_out", suffix_out)

    v_t = model.base_model.action_out_proj(suffix_out[:, -model.action_horizon :])
    v_t = _tap_debug_tensor(debug_tensors, component_scales, "v_t", v_t)
    chunked_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
    return DebugForwardResult(chunked_loss, debug_tensors, _PI05_WITH_SIDENET_COMPONENT_NAMES)


def _debug_forward(
    model: _model.BaseModel,
    rng: jax.Array,
    observation: _model.Observation,
    actions: _model.Actions,
    *,
    train: bool,
    component_scales: dict[str, jax.Array] | None = None,
) -> DebugForwardResult:
    if _pi05_with_sidenet is not None and isinstance(model, _pi05_with_sidenet.Pi05WithSideNet):
        return _pi05_with_sidenet_debug_forward(
            model,
            rng,
            observation,
            actions,
            train=train,
            component_scales=component_scales,
        )
    if isinstance(model, _pi0.Pi0):
        return _pi0_debug_forward(
            model,
            rng,
            observation,
            actions,
            train=train,
            component_scales=component_scales,
        )
    return DebugForwardResult(
        model.compute_loss(rng, observation, actions, train=train),
        {},
        (),
    )


class TrainingChecker:
    def __init__(self, config: TrainingCheckerConfig, output_dir: pathlib.Path):
        self._config = config
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self._output_dir / "reports.jsonl"

    def should_run(self, step: int) -> bool:
        if step < self._config.initial_consecutive_steps:
            return True
        return step % self._config.interval == 0

    def _compute_loss_and_grads(
        self,
        model: _model.BaseModel,
        train_config: Any,
        rng: jax.Array,
        observation: _model.Observation,
        actions: _model.Actions,
    ) -> tuple[float, jax.Array, dict[str, jax.Array], nnx.State, tuple[str, ...]]:
        diff_state = nnx.DiffState(0, train_config.trainable_filter)

        def loss_fn(
            model: _model.BaseModel,
            rng: jax.Array,
            observation: _model.Observation,
            actions: _model.Actions,
        ):
            result = _debug_forward(model, rng, observation, actions, train=True)
            per_sample_loss = _reduce_per_sample_loss(result.chunked_loss)
            return jnp.mean(per_sample_loss), (per_sample_loss, result.debug_tensors, result.component_names)

        (loss, (per_sample_loss, debug_tensors, component_names)), grads = nnx.value_and_grad(
            loss_fn,
            argnums=diff_state,
            has_aux=True,
        )(model, rng, observation, actions)
        return _to_float(loss), per_sample_loss, debug_tensors, grads, component_names

    def _per_sample_gradient_report(
        self,
        state: Any,
        train_config: Any,
        rng: jax.Array,
        observation: _model.Observation,
        actions: _model.Actions,
    ) -> list[dict[str, Any]]:
        batch_size = int(actions.shape[0])
        report = []
        diff_state = nnx.DiffState(0, train_config.trainable_filter)

        def sample_loss_fn(
            model: _model.BaseModel,
            rng: jax.Array,
            observation: _model.Observation,
            actions: _model.Actions,
        ):
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            per_sample_loss = _reduce_per_sample_loss(chunked_loss)
            return jnp.mean(per_sample_loss)

        for sample_index in range(min(batch_size, self._config.max_per_sample)):
            sample_model = nnx.merge(state.model_def, state.params)
            sample_model.train()
            sample_obs = _slice_batch(observation, sample_index)
            sample_actions = _slice_batch(actions, sample_index)
            sample_rng = jax.random.fold_in(rng, sample_index)
            sample_loss, sample_grads = nnx.value_and_grad(sample_loss_fn, argnums=diff_state)(
                sample_model,
                sample_rng,
                sample_obs,
                sample_actions,
            )
            flat_sample_grads = _tree_to_flat_arrays(sample_grads)
            report.append(
                {
                    "index": sample_index,
                    "loss": _to_float(sample_loss),
                    "grad_norm": _to_float(optax.global_norm(sample_grads)),
                    "module_grad_norms": _aggregate_norms(flat_sample_grads),
                }
            )
        return report

    def _component_ablation_report(
        self,
        model: _model.BaseModel,
        component_names: tuple[str, ...],
        rng: jax.Array,
        observation: _model.Observation,
        actions: _model.Actions,
        reference_loss: float,
    ) -> dict[str, float]:
        if not component_names:
            return {}

        report: dict[str, float] = {}
        for name in component_names[: self._config.max_component_paths]:
            scales = {name: jnp.asarray(0.0, dtype=jnp.float32)}
            result = _debug_forward(
                model,
                rng,
                observation,
                actions,
                train=True,
                component_scales=scales,
            )
            ablated_loss = _to_float(jnp.mean(_reduce_per_sample_loss(result.chunked_loss)))
            report[name] = ablated_loss - reference_loss
        return report

    def _component_scale_gradient_report(
        self,
        model: _model.BaseModel,
        component_names: tuple[str, ...],
        rng: jax.Array,
        observation: _model.Observation,
        actions: _model.Actions,
    ) -> dict[str, float]:
        if not component_names:
            return {}

        component_names = component_names[: self._config.max_component_paths]
        scale_tree = {name: jnp.asarray(1.0, dtype=jnp.float32) for name in component_names}

        def loss_from_scales(component_scales: dict[str, jax.Array]) -> jax.Array:
            result = _debug_forward(
                model,
                rng,
                observation,
                actions,
                train=True,
                component_scales=component_scales,
            )
            return jnp.mean(_reduce_per_sample_loss(result.chunked_loss))

        scale_grads = jax.grad(loss_from_scales)(scale_tree)
        return {name: _to_float(value) for name, value in scale_grads.items()}

    def run(
        self,
        train_config: Any,
        rng: jax.Array,
        state: Any,
        batch: tuple[_model.Observation, _model.Actions],
    ) -> dict[str, Any]:
        model = nnx.merge(state.model_def, state.params)
        model.train()

        observation, actions = batch
        checker_rng = jax.random.fold_in(rng, state.step)
        loss, per_sample_loss, debug_tensors, grads, component_names = self._compute_loss_and_grads(
            model,
            train_config,
            checker_rng,
            observation,
            actions,
        )

        trainable_params = state.params.filter(train_config.trainable_filter)
        updates, _ = state.tx.update(grads, state.opt_state, trainable_params)

        flat_params = _tree_to_flat_arrays(trainable_params)
        flat_grads = _tree_to_flat_arrays(grads)
        flat_updates = _tree_to_flat_arrays(updates)

        report: dict[str, Any] = {
            "step": int(np.asarray(jax.device_get(state.step))),
            "batch": {
                "size": int(actions.shape[0]),
                "loss_mean": loss,
                "loss_std": _to_float(jnp.std(per_sample_loss)),
                "loss_min": _to_float(jnp.min(per_sample_loss)),
                "loss_max": _to_float(jnp.max(per_sample_loss)),
                "loss_range": _to_float(jnp.max(per_sample_loss) - jnp.min(per_sample_loss)),
            },
            "global": {
                "grad_norm": _to_float(optax.global_norm(grads)),
                "param_norm": _to_float(optax.global_norm(trainable_params)),
                "update_norm": _to_float(optax.global_norm(updates)),
            },
            "gradients": {
                "module_grad_norms": _aggregate_norms(flat_grads),
                "top_leaf_grad_norms": _topk_leaf_norms(flat_grads, self._config.max_leaf_paths),
            },
            "parameters": {
                "module_param_norms": _aggregate_norms(flat_params),
                "top_leaf_param_norms": _topk_leaf_norms(flat_params, self._config.max_leaf_paths),
            },
            "updates": {
                "module_update_norms": _aggregate_norms(flat_updates),
                "top_leaf_update_norms": _topk_leaf_norms(flat_updates, self._config.max_leaf_paths),
            },
            "activations": {name: _tensor_stats(value) for name, value in debug_tensors.items()},
        }

        if self._config.log_full_per_sample_losses:
            report["batch"]["per_sample_losses"] = _to_python(per_sample_loss)

        if "side_tokens" in debug_tensors and "suffix_tokens_pre_side" in debug_tensors:
            side_norm = _to_float(_array_l2_norm(debug_tensors["side_tokens"]))
            suffix_norm = _to_float(_array_l2_norm(debug_tensors["suffix_tokens_pre_side"]))
            report.setdefault("ratios", {})["side_to_suffix_pre"] = side_norm / max(suffix_norm, 1e-8)
        if "side_tokens" in debug_tensors and "prefix_tokens" in debug_tensors:
            side_norm = _to_float(_array_l2_norm(debug_tensors["side_tokens"]))
            prefix_norm = _to_float(_array_l2_norm(debug_tensors["prefix_tokens"]))
            report.setdefault("ratios", {})["side_to_prefix"] = side_norm / max(prefix_norm, 1e-8)

        if self._config.compute_component_ablation:
            report["component_ablation_delta_loss"] = self._component_ablation_report(
                model,
                component_names,
                checker_rng,
                observation,
                actions,
                loss,
            )

        if self._config.compute_component_scale_gradients:
            report["component_scale_gradients"] = self._component_scale_gradient_report(
                model,
                component_names,
                checker_rng,
                observation,
                actions,
            )

        if self._config.compute_per_sample_gradients:
            report["per_sample"] = self._per_sample_gradient_report(
                state,
                train_config,
                checker_rng,
                observation,
                actions,
            )

        return _to_python(report)

    def summarize_scalars(self, report: dict[str, Any]) -> dict[str, float]:
        summary = {
            "train_checker/loss_mean": float(report["batch"]["loss_mean"]),
            "train_checker/loss_std": float(report["batch"]["loss_std"]),
            "train_checker/loss_range": float(report["batch"]["loss_range"]),
            "train_checker/grad_norm": float(report["global"]["grad_norm"]),
            "train_checker/update_norm": float(report["global"]["update_norm"]),
        }
        for ratio_name, ratio_value in report.get("ratios", {}).items():
            summary[f"train_checker/{ratio_name}"] = float(ratio_value)
        for name, value in list(report.get("component_ablation_delta_loss", {}).items())[:4]:
            summary[f"train_checker/ablation/{name.replace('/', '.')}"] = float(value)
        for name, value in list(report.get("component_scale_gradients", {}).items())[:4]:
            summary[f"train_checker/scale_grad/{name.replace('/', '.')}"] = float(value)
        return summary

    def write(self, step: int, report: dict[str, Any]) -> pathlib.Path:
        path = self._output_dir / f"step_{step:08d}.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        with self._jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")
        latest_path = self._output_dir / "latest.json"
        latest_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        logger.info("Wrote training checker report to %s", path)
        return path
