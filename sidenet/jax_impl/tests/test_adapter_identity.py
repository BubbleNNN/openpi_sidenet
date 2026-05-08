"""Sanity test: strict LlamaAdapter wrapper's first forward == base Pi0's forward.

Run with ``pytest sidenet/jax_impl/tests/test_adapter_identity.py -s`` on a box
that can import jax + openpi. This test is the primary guarantee that the
adapter wrapper's first training step produces the exact pretrained loss.

It creates a fake observation, builds both a base ``Pi0`` and a
``Pi05WithSideNetAdapter`` from the same RNG seed, copies the base model's
LLM params into the adapter model (via the scan-split helper), and checks
that the suffix-output tensor is numerically identical up to fp tolerance.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import pytest

from openpi.models import pi0 as _pi0
from openpi.models import pi0_config
from openpi.models import model as _model
from sidenet.jax_impl import pi05_with_sidenet_llama_adapter as _adapter
from sidenet.jax_impl import weight_loader as _wl


def _make_fake_observation(config):
    obs, _ = config.inputs_spec(batch_size=1)
    def to_zeros(spec):
        if spec is None:
            return None
        return jnp.zeros(spec.shape, dtype=spec.dtype)
    obs = jax.tree.map(to_zeros, obs, is_leaf=lambda x: x is None or hasattr(x, "shape"))
    return dataclasses.replace(
        obs,
        image_masks={key: jnp.ones_like(value) for key, value in obs.image_masks.items()},
        tokenized_prompt_mask=(
            jnp.ones_like(obs.tokenized_prompt_mask)
            if obs.tokenized_prompt_mask is not None
            else None
        ),
    )


def test_split_scanned_layers_for_adapter_uses_last_layers():
    params = {
        "PaliGemma": {
            "llm": {
                "layers": {
                    "probe": jnp.arange(4),
                },
            },
            "img": {
                "layers": {
                    "probe": jnp.arange(4),
                },
            },
        },
    }

    split = _wl.split_scanned_layers_for_adapter(params, num_injected_layers=1)

    np.testing.assert_array_equal(split["PaliGemma"]["llm"]["base_layers"]["probe"], np.array([0, 1, 2]))
    np.testing.assert_array_equal(split["PaliGemma"]["llm"]["injected_layers"]["probe"], np.array([3]))
    np.testing.assert_array_equal(split["PaliGemma"]["img"]["layers"]["probe"], np.array([0, 1, 2, 3]))


def test_split_scanned_layers_all_base_when_zero_injected_layers():
    params = {
        "PaliGemma": {
            "llm": {
                "layers": {
                    "probe": jnp.arange(4),
                },
            },
        },
    }

    split = _wl.split_scanned_layers_for_adapter(params, num_injected_layers=0)

    np.testing.assert_array_equal(split["PaliGemma"]["llm"]["base_layers"]["probe"], np.array([0, 1, 2, 3]))
    assert "injected_layers" not in split["PaliGemma"]["llm"]


def test_merge_loaded_params_raises_on_shape_mismatch():
    loaded = {"PaliGemma": {"llm": {"probe": jnp.zeros((2, 3))}}}
    reference = {"PaliGemma": {"llm": {"probe": jnp.zeros((2, 4))}}}

    with pytest.raises(ValueError, match="shape-mismatched"):
        _wl.merge_loaded_params(loaded, reference, log_prefix="test_mismatch")


def test_first_step_matches_base():
    # A *tiny* model to keep the test cheap; the identity property is not
    # size-specific.
    base_cfg = pi0_config.Pi0Config(
        pi05=True,
        dtype="float32",
        paligemma_variant="smoke_paligemma",
        action_expert_variant="smoke_action_expert",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
    )
    adapter_cfg = _adapter.Pi05WithSideNetAdapterConfig(
        pi05=True,
        dtype="float32",
        paligemma_variant="smoke_paligemma",
        action_expert_variant="smoke_action_expert",
        action_dim=8,
        action_horizon=4,
        max_token_len=16,
        num_injected_layers=1,  # smoke depth is 2, so the last layer is injected
        sidenet_config_path="./sidenet/sidenet_smoke_config.yaml",
    )

    rng = jax.random.key(0)
    base = base_cfg.create(rng)
    adapter = adapter_cfg.create(rng)

    # Copy LLM params from base into adapter: flatten base params, split the
    # scanned `layers/*` entries, then merge into the adapter's init params.
    _, base_state = nnx.split(base)
    base_params = base_state.to_pure_dict()
    # Only take the LLM + image encoder + action projections -- everything the
    # adapter model also has under the same names.
    split_params = _wl.split_scanned_layers_for_adapter(
        base_params, num_injected_layers=adapter_cfg.num_injected_layers
    )

    graphdef, state = nnx.split(adapter)
    ref = state.to_pure_dict()
    merged = _wl.merge_loaded_params(split_params, ref, log_prefix="test_identity")
    state.replace_by_pure_dict(merged)
    adapter = nnx.merge(graphdef, state)

    obs = _make_fake_observation(adapter_cfg)
    actions = jnp.zeros((1, base_cfg.action_horizon, base_cfg.action_dim), dtype=jnp.float32)

    base_loss = base.compute_loss(jax.random.key(42), obs, actions, train=False)
    adapter_loss = adapter.compute_loss(jax.random.key(42), obs, actions, train=False)

    diff = jnp.abs(base_loss - adapter_loss).max()
    print(f"base_loss={np.array(base_loss)} adapter_loss={np.array(adapter_loss)} diff={diff}")
    assert diff < 1e-4, (
        f"Adapter first-step loss must match base at init (gate=0). diff={diff}"
    )


if __name__ == "__main__":
    test_first_step_matches_base()
