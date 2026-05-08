"""Weight-loading helpers for the standalone JAX SideNet wrapper.

The base OpenPI `CheckpointWeightLoader` only fills missing LoRA parameters.
That is too strict for additive wrappers like `Pi05WithSideNet`, where the
checkpoint may only contain the base PI0.5 parameters and the SideNet
parameters should stay at their freshly initialized values.

This module also provides ``split_scanned_layers_for_adapter`` for the
LlamaAdapter-style wrapper, which replaces the single scanned ``layers`` Gemma
param tree with two separate scans (early ``base_layers`` + final
``injected_layers``).
"""

from __future__ import annotations

import dataclasses
import logging

import flax.traverse_util
import numpy as np

from openpi.models import model as _model
from openpi.shared import array_typing as at

logger = logging.getLogger(__name__)


def split_scanned_layers_for_adapter(
    params: at.Params,
    *,
    num_injected_layers: int,
    layers_key: str = "layers",
    base_key: str = "base_layers",
    injected_key: str = "injected_layers",
    target_path: tuple[str, ...] = ("PaliGemma", "llm"),
) -> at.Params:
    """Split a base-Pi0 scanned ``layers/...`` sub-tree into base + injected.

    A base Pi0 checkpoint stores all ``depth`` transformer blocks as a single
    ``nn.scan``-packed tree rooted at ``.../llm/layers/...`` whose leaves all
    have a leading axis of length ``depth``. The injection wrapper inserts
    side hidden states at the *last* ``num_injected_layers`` layers, so
    ``base_layers`` receives the first ``depth - num_injected_layers``
    slices and ``injected_layers`` receives the remaining ``num_injected_layers``
    slices. This helper performs that split in-place on every matching
    sub-tree so the checkpoint can be fed unchanged into ``merge_loaded_params``.

    - If the checkpoint already uses the split layout (i.e. has a
      ``base_layers`` or ``injected_layers`` tree), it is returned unchanged.
    - If the checkpoint has ``layers`` but the leading axis is *smaller*
      than ``num_injected_layers``, a ``ValueError`` is raised so callers
      do not silently get truncated splits.
    """
    if num_injected_layers < 0:
        raise ValueError(f"num_injected_layers must be non-negative, got {num_injected_layers}.")

    flat = flax.traverse_util.flatten_dict(params, sep="/")

    def find_target_layers_segment(key: str) -> int | None:
        parts = key.split("/")
        needle = (*target_path, layers_key)
        for idx in range(len(parts) - len(needle) + 1):
            if tuple(parts[idx : idx + len(needle)]) == needle:
                return idx + len(target_path)
        return None

    def replace_target_layers_segment(key: str, new_key: str) -> str:
        parts = key.split("/")
        layers_idx = find_target_layers_segment(key)
        if layers_idx is None:
            raise ValueError(f"`{key}` is not under {'/'.join((*target_path, layers_key))}.")
        parts[layers_idx] = new_key
        return "/".join(parts)

    # Detect only the Gemma LLM scan tree. Other modules may also use a
    # ``layers`` segment and must not be split into adapter/base scans.
    keys_to_split: list[str] = [
        k for k in flat
        if find_target_layers_segment(k) is not None
        and f"/{base_key}/" not in f"/{k}/"
        and f"/{injected_key}/" not in f"/{k}/"
    ]
    if not keys_to_split:
        return params

    result: dict[str, object] = {k: v for k, v in flat.items() if k not in keys_to_split}
    split_count = 0
    for key in keys_to_split:
        value = flat[key]
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) == 0:
            # Scalar -- leave alone (unlikely for Gemma params).
            result[key] = value
            continue
        depth = shape[0]
        if depth <= num_injected_layers:
            raise ValueError(
                f"Cannot split `{key}` with leading axis {depth} into "
                f"base ({depth - num_injected_layers}) + injected "
                f"({num_injected_layers}) chunks."
            )
        base_value = value[: depth - num_injected_layers]
        base_new_key = replace_target_layers_segment(key, base_key)
        result[base_new_key] = base_value
        if num_injected_layers > 0:
            injected_value = value[depth - num_injected_layers :]
            injected_new_key = replace_target_layers_segment(key, injected_key)
            result[injected_new_key] = injected_value
        split_count += 1

    logger.info(
        "split_scanned_layers_for_adapter: split %d scanned `%s/*` entries "
        "into `%s/*` (first depth-%d slices) + `%s/*` (last %d slices).",
        split_count,
        layers_key,
        base_key,
        num_injected_layers,
        injected_key,
        num_injected_layers,
    )
    return flax.traverse_util.unflatten_dict(result, sep="/")


def merge_loaded_params(
    loaded_params: at.Params,
    reference_params: at.Params,
    *,
    log_prefix: str = "sidenet_jax",
) -> at.Params:
    """Merge a possibly-partial checkpoint into an initialized parameter tree."""
    flat_ref = flax.traverse_util.flatten_dict(reference_params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    result: dict[str, object] = {}
    matched = 0
    missing = 0
    shape_mismatches: list[tuple[str, object, object]] = []

    for key, ref_value in flat_ref.items():
        loaded_value = flat_loaded.get(key)
        if loaded_value is None:
            result[key] = ref_value
            missing += 1
            continue

        loaded_shape = getattr(loaded_value, "shape", None)
        ref_shape = getattr(ref_value, "shape", None)
        if loaded_shape != ref_shape:
            shape_mismatches.append((key, loaded_shape, ref_shape))
            continue

        loaded_dtype = getattr(loaded_value, "dtype", None)
        ref_dtype = getattr(ref_value, "dtype", None)
        if loaded_dtype != ref_dtype and hasattr(loaded_value, "astype"):
            loaded_value = loaded_value.astype(ref_dtype)

        result[key] = loaded_value
        matched += 1

    if shape_mismatches:
        preview = "\n".join(
            f"  - {key}: loaded={loaded_shape}, reference={ref_shape}"
            for key, loaded_shape, ref_shape in shape_mismatches[:10]
        )
        suffix = (
            f"\n  ... and {len(shape_mismatches) - 10} more"
            if len(shape_mismatches) > 10
            else ""
        )
        raise ValueError(
            f"{log_prefix} found {len(shape_mismatches)} shape-mismatched "
            f"checkpoint parameter(s):\n{preview}{suffix}"
        )

    extra_keys = sorted(set(flat_loaded) - set(flat_ref))
    logger.info(
        "%s weight merge summary: matched=%d missing=%d shape_mismatch=0 extra=%d",
        log_prefix,
        matched,
        missing,
        len(extra_keys),
    )
    if extra_keys:
        preview = ", ".join(extra_keys[:10])
        logger.info("%s ignored extra checkpoint keys: %s", log_prefix, preview)

    return flax.traverse_util.unflatten_dict(result, sep="/")


@dataclasses.dataclass(frozen=True)
class PartialCheckpointWeightLoader:
    """Load a checkpoint into a larger initialized parameter tree."""

    params_path: str
    log_prefix: str = "sidenet_jax"

    def load(self, params: at.Params) -> at.Params:
        import openpi.shared.download as download

        loaded_params = _model.restore_params(
            download.maybe_download(self.params_path),
            restore_type=np.ndarray,
        )
        return merge_loaded_params(loaded_params, params, log_prefix=self.log_prefix)


@dataclasses.dataclass(frozen=True)
class AdapterCheckpointWeightLoader:
    """Weight loader for the LlamaAdapter variant.

    Like ``PartialCheckpointWeightLoader`` but additionally splits the base
    checkpoint's single ``layers/*`` scan tree into early ``base_layers/*``
    + final ``injected_layers/*`` before merging, so the scanned params land
    in the correct slots of ``InjectionModule``.
    """

    params_path: str
    num_injected_layers: int = 5
    log_prefix: str = "sidenet_adapter"

    def load(self, params: at.Params) -> at.Params:
        import openpi.shared.download as download

        loaded_params = _model.restore_params(
            download.maybe_download(self.params_path),
            restore_type=np.ndarray,
        )
        loaded_params = split_scanned_layers_for_adapter(
            loaded_params, num_injected_layers=self.num_injected_layers,
        )
        return merge_loaded_params(loaded_params, params, log_prefix=self.log_prefix)
