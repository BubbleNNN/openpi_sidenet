"""Standalone JAX PI0.5 + SideNet wrapper.

This keeps the SideNet integration local to ``sidenet/jax`` by composing the
base JAX ``Pi0`` model instead of modifying ``src/openpi/models/pi0.py``.

Current scope:
- only ``pi05=True`` is supported
- only single-frame ``ft_sensor`` is wired into SideNet
- text conditioning comes from contextualized VLM prefix hidden states
"""

from __future__ import annotations

import dataclasses

import einops
from flax import nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config
from openpi.shared import array_typing as at

from sidenet.jax.sidenet import SideNet
from sidenet.jax import weight_loader as sidenet_weight_loader
from sidenet.parse_config import load_sidenet_config


@dataclasses.dataclass(frozen=True)
class Pi05WithSideNetConfig(pi0_config.Pi0Config):
    """Config for the standalone JAX PI0.5 + SideNet wrapper."""

    sidenet_config_path: str = "./sidenet/sidenet_config.yaml"
    ft_sensor_dim: int = 12
    pi05: bool = True

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05:
            raise ValueError("Pi05WithSideNetConfig only supports pi05=True.")

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi05WithSideNet":
        return Pi05WithSideNet(self, rngs=nnx.Rngs(rng))

    @override
    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "Pi05WithSideNet":
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        reference_params = state.to_pure_dict()
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(reference_params, params)
        params = sidenet_weight_loader.merge_loaded_params(
            params,
            reference_params,
            log_prefix="pi05_with_sidenet_jax.load",
        )
        at.check_pytree_equality(expected=reference_params, got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        observation_spec, action_spec = super().inputs_spec(batch_size=batch_size)
        ft_sensor_spec = jax.ShapeDtypeStruct([batch_size, self.ft_sensor_dim], jnp.float32)
        with at.disable_typechecking():
            observation_spec = dataclasses.replace(
                observation_spec,
                ft_sensor=ft_sensor_spec,
            )
        return observation_spec, action_spec


class Pi05WithSideNet(_model.BaseModel):
    """Standalone JAX PI0.5 wrapper with SideNet suffix-token injection."""

    def __init__(self, config: Pi05WithSideNetConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)

        self.base_model = _pi0.Pi0(config, rngs=rngs)
        self.ft_sensor_dim = config.ft_sensor_dim

        sidenet_cfg = load_sidenet_config(config.sidenet_config_path)
        self.sidenet = nnx_bridge.ToNNX(
            SideNet(
                branch_input_dims=(("ft_sensor", config.ft_sensor_dim),),
                d_model=sidenet_cfg.d_model,
                num_perceiver_queries=sidenet_cfg.num_perceiver_queries,
                num_input_tokens=sidenet_cfg.num_input_tokens,
                num_perceiver_layers=sidenet_cfg.num_perceiver_layers,
                num_fusion_queries=sidenet_cfg.num_fusion_queries,
                num_heads=sidenet_cfg.num_heads,
                text_embed_dim=sidenet_cfg.text_embed_dim,
                output_hidden_features=sidenet_cfg.output_hidden_features,
                output_dim=sidenet_cfg.output_dim,
            )
        )

        fake_modality_inputs = {
            "ft_sensor": jnp.ones((1, config.ft_sensor_dim), dtype=jnp.float32),
        }
        fake_text_embeddings = jnp.ones(
            (1, config.max_token_len, sidenet_cfg.text_embed_dim),
            dtype=jnp.float32,
        )
        fake_text_mask = jnp.ones((1, config.max_token_len), dtype=jnp.bool_)
        self.sidenet.lazy_init(
            fake_modality_inputs,
            fake_text_embeddings,
            fake_text_mask,
            deterministic=True,
            rngs=rngs,
        )

    def _resolve_ft_sensor(self, observation: _model.Observation) -> jnp.ndarray:
        ft_sensor = observation.ft_sensor
        if ft_sensor is None and observation.modalities is not None:
            ft_sensor = observation.modalities.get("ft_sensor")
        if ft_sensor is None:
            raise ValueError(
                "Pi05WithSideNet requires `observation.ft_sensor` "
                "or `observation.modalities[\"ft_sensor\"]`."
            )

        ft_sensor = jnp.asarray(ft_sensor, dtype=jnp.float32)
        if ft_sensor.shape[-1] != self.ft_sensor_dim:
            raise ValueError(
                f"Expected ft_sensor feature dim {self.ft_sensor_dim}, got {ft_sensor.shape[-1]}."
            )
        if ft_sensor.ndim == 2:
            return ft_sensor
        if ft_sensor.ndim == 3 and ft_sensor.shape[-2] == 1:
            return jnp.squeeze(ft_sensor, axis=-2)
        raise ValueError(
            "SideNet JAX wrapper only supports single-frame ft_sensor with shape "
            "`(B, 12)` or `(B, 1, 12)`."
        )

    def _run_prefix_vlm(
        self,
        prefix_tokens: jnp.ndarray,
        prefix_mask: jnp.ndarray,
        prefix_ar_mask: jnp.ndarray,
        *,
        deterministic: bool,
    ) -> tuple[jnp.ndarray, object]:
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.base_model.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
            deterministic=deterministic,
        )
        return prefix_out, kv_cache

    def _extract_text_features(
        self,
        prefix_out: jnp.ndarray,
        observation: _model.Observation,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            raise ValueError("Pi05WithSideNet requires tokenized prompt inputs for text conditioning.")

        text_len = observation.tokenized_prompt.shape[1]
        text_features = prefix_out[:, -text_len:, :]
        return text_features, observation.tokenized_prompt_mask

    def _run_sidenet(
        self,
        ft_sensor: jnp.ndarray,
        text_features: jnp.ndarray,
        text_mask: jnp.ndarray,
        *,
        deterministic: bool,
    ) -> jnp.ndarray:
        # SideNet owns the text projector; the wrapper is responsible for
        # sourcing contextualized VLM text states and the single-frame ft input.
        modality_inputs = {
            "ft_sensor": ft_sensor,
        }
        return self.sidenet(
            modality_inputs,
            text_features,
            text_mask,
            deterministic=deterministic,
        )

    @staticmethod
    def _prepend_side_tokens(
        side_tokens: jnp.ndarray,
        suffix_tokens: jnp.ndarray,
        suffix_mask: jnp.ndarray,
        suffix_ar_mask: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if side_tokens.shape[-1] != suffix_tokens.shape[-1]:
            raise ValueError(
                f"SideNet output dim ({side_tokens.shape[-1]}) must match "
                f"suffix width ({suffix_tokens.shape[-1]})."
            )

        side_tokens = side_tokens.astype(suffix_tokens.dtype)
        side_mask = jnp.ones(side_tokens.shape[:2], dtype=suffix_mask.dtype)
        side_ar_mask = jnp.zeros((side_tokens.shape[1],), dtype=suffix_ar_mask.dtype)
        side_ar_mask = side_ar_mask.at[0].set(True)

        suffix_tokens = jnp.concatenate([side_tokens, suffix_tokens], axis=1)
        suffix_mask = jnp.concatenate([side_mask, suffix_mask], axis=1)
        suffix_ar_mask = jnp.concatenate([side_ar_mask, suffix_ar_mask], axis=0)
        return suffix_tokens, suffix_mask, suffix_ar_mask

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.base_model.embed_prefix(observation)
        prefix_out, _ = self._run_prefix_vlm(
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
            deterministic=not train,
        )
        text_features, text_mask = self._extract_text_features(prefix_out, observation)
        ft_sensor = self._resolve_ft_sensor(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.base_model.embed_suffix(observation, x_t, time)

        side_tokens = self._run_sidenet(
            ft_sensor,
            text_features,
            text_mask,
            deterministic=not train,
        )
        suffix_tokens, suffix_mask, suffix_ar_mask = self._prepend_side_tokens(
            side_tokens,
            suffix_tokens,
            suffix_mask,
            suffix_ar_mask,
        )

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.base_model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.base_model.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.base_model.embed_prefix(observation)
        prefix_out, kv_cache = self._run_prefix_vlm(
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
            deterministic=True,
        )
        text_features, text_mask = self._extract_text_features(prefix_out, observation)
        ft_sensor = self._resolve_ft_sensor(observation)
        side_tokens = self._run_sidenet(
            ft_sensor,
            text_features,
            text_mask,
            deterministic=True,
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.base_model.embed_suffix(
                observation,
                x_t,
                jnp.broadcast_to(time, batch_size),
            )
            suffix_tokens, suffix_mask, suffix_ar_mask = self._prepend_side_tokens(
                side_tokens,
                suffix_tokens,
                suffix_mask,
                suffix_ar_mask,
            )

            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (_, suffix_out), _ = self.base_model.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.base_model.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
