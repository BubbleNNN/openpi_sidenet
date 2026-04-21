"""Weight-loading helpers for the standalone JAX SideNet wrapper.

The base OpenPI `CheckpointWeightLoader` only fills missing LoRA parameters.
That is too strict for additive wrappers like `Pi05WithSideNet`, where the
checkpoint may only contain the base PI0.5 parameters and the SideNet
parameters should stay at their freshly initialized values.
"""

from __future__ import annotations

import dataclasses
import logging

import flax.traverse_util
import numpy as np

from openpi.models import model as _model
from openpi.shared import array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


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
            result[key] = ref_value
            continue

        loaded_dtype = getattr(loaded_value, "dtype", None)
        ref_dtype = getattr(ref_value, "dtype", None)
        if loaded_dtype != ref_dtype and hasattr(loaded_value, "astype"):
            loaded_value = loaded_value.astype(ref_dtype)

        result[key] = loaded_value
        matched += 1

    extra_keys = sorted(set(flat_loaded) - set(flat_ref))
    logger.info(
        "%s weight merge summary: matched=%d missing=%d shape_mismatch=%d extra=%d",
        log_prefix,
        matched,
        missing,
        len(shape_mismatches),
        len(extra_keys),
    )
    for key, loaded_shape, ref_shape in shape_mismatches[:10]:
        logger.warning(
            "%s skipped shape-mismatched param `%s`: loaded=%s, reference=%s",
            log_prefix,
            key,
            loaded_shape,
            ref_shape,
        )
    if len(shape_mismatches) > 10:
        logger.warning("%s skipped %d additional shape-mismatched params", log_prefix, len(shape_mismatches) - 10)
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
        loaded_params = _model.restore_params(
            download.maybe_download(self.params_path),
            restore_type=np.ndarray,
        )
        return merge_loaded_params(loaded_params, params, log_prefix=self.log_prefix)
