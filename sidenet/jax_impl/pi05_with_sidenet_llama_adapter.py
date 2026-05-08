"""Strict Pi0.5 + SideNet wrapper with split-softmax multimodal injection.

This wrapper replaces the standard ``gemma.Module`` used inside Pi0 with
a custom ``adapter_gemma.InjectionModule`` in which the *last*
``num_injected_layers`` transformer blocks inject SideNet-produced multimodal
hidden states via split softmax and zero-initialised per-head gates.

Design guarantees:

* **Identity at init** -- ``injection_gate`` is zero-initialised, so
  ``tanh(0) = 0`` makes the side-attention branch contribute exactly 0
  at step 0. When the pretrained pi0.5 weights are loaded into the
  non-adapter parameters, the *first* forward/backward pass produces
  exactly the pretrained pi0.5 loss (bit-for-bit up to unrelated
  numerical differences like dropout RNG).
* **Weight compatibility** -- the base ``layers`` scan parameter (shape
  ``(depth, ...)``) in a base pi0.5 checkpoint is split into
  ``base_layers`` (first ``depth - num_injected_layers`` slices) and
  ``injected_layers`` (last ``num_injected_layers`` slices). See
  ``sidenet.jax_impl.weight_loader.split_scanned_layers_for_adapter``.
"""

from __future__ import annotations

import dataclasses
import logging

import einops
from flax import nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from typing_extensions import override

from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config
from openpi.models import siglip as _siglip
from openpi.shared import array_typing as at

from sidenet.jax_impl import adapter_gemma
from sidenet.jax_impl import weight_loader as sidenet_weight_loader
from sidenet.jax_impl.sidenet import SideNet
from sidenet.parse_config import load_sidenet_config

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Pi05WithSideNetAdapterConfig(pi0_config.Pi0Config):
    """Config for strict LlamaAdapter-style PI0.5 + SideNet wrapper."""

    sidenet_config_path: str = "./sidenet/sidenet_config.yaml"
    ft_sensor_dim: int = 12
    num_injected_layers: int = 5
    pi05: bool = True

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05:
            raise ValueError("Pi05WithSideNetAdapterConfig only supports pi05=True.")
        if self.num_injected_layers < 0:
            raise ValueError(
                f"num_injected_layers must be non-negative, got {self.num_injected_layers}."
            )

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi05WithSideNetAdapter":
        return Pi05WithSideNetAdapter(self, rngs=nnx.Rngs(rng))

    @override
    def load(
        self, params: at.Params, *, remove_extra_params: bool = True
    ) -> "Pi05WithSideNetAdapter":
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        reference_params = state.to_pure_dict()

        # If the loaded checkpoint was produced by the base Pi0 model, its LLM
        # layers are a single scanned param tree with leading axis = depth.
        # Split it into injected_layers (last N) and base_layers (remaining)
        # so it matches the InjectionModule structure.
        params = sidenet_weight_loader.split_scanned_layers_for_adapter(
            params,
            num_injected_layers=self.num_injected_layers,
        )

        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(reference_params, params)
        params = sidenet_weight_loader.merge_loaded_params(
            params,
            reference_params,
            log_prefix="pi05_with_sidenet_llama_adapter.load",
        )
        at.check_pytree_equality(
            expected=reference_params, got=params, check_shapes=True, check_dtypes=False
        )
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    @override
    def inputs_spec(
        self, *, batch_size: int = 1
    ) -> tuple[_model.Observation, _model.Actions]:
        observation_spec, action_spec = super().inputs_spec(batch_size=batch_size)
        if self.num_injected_layers == 0:
            return observation_spec, action_spec
        ft_sensor_spec = jax.ShapeDtypeStruct(
            [batch_size, self.ft_sensor_dim], jnp.float32
        )
        with at.disable_typechecking():
            observation_spec = dataclasses.replace(observation_spec, ft_sensor=ft_sensor_spec)
        return observation_spec, action_spec


class Pi05WithSideNetAdapter(_model.BaseModel):
    """Pi0.5 with SideNet split-softmax injection in the last action layers.

    Structurally a drop-in replacement for ``_pi0.Pi0`` whose LLM is a
    custom ``adapter_gemma.InjectionModule`` rather than ``_gemma.Module``.
    All non-LLM modules (image encoder, action/time projections) are
    identical to ``_pi0.Pi0`` so that pretrained weights transfer
    straight across.
    """

    def __init__(self, config: Pi05WithSideNetAdapterConfig, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        sidenet_cfg = load_sidenet_config(config.sidenet_config_path)
        if config.num_injected_layers == 0:
            logger.warning(
                "Pi05WithSideNetAdapter initialized with num_injected_layers=0; "
                "SideNet will not be created and no injection will be applied."
            )
        else:
            logger.info(
                "Pi05WithSideNetAdapter initialized with num_injected_layers=%d.",
                config.num_injected_layers,
            )
        llm = nnx_bridge.ToNNX(
            adapter_gemma.InjectionModule(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
                num_injected_layers=config.num_injected_layers,
                injection_expert_idx=1,
                num_injection_tokens=sidenet_cfg.num_injection_tokens,
                fused_memory_dim=sidenet_cfg.action_expert_hidden_dim,
            )
        )
        llm.lazy_init(
            rngs=rngs,
            method="init",
            use_adarms=[False, True] if config.pi05 else [False, False],
        )

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(
                action_expert_config.width, action_expert_config.width, rngs=rngs
            )
            self.time_mlp_out = nnx.Linear(
                action_expert_config.width, action_expert_config.width, rngs=rngs
            )
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(
                2 * action_expert_config.width, action_expert_config.width, rngs=rngs
            )
            self.action_time_mlp_out = nnx.Linear(
                action_expert_config.width, action_expert_config.width, rngs=rngs
            )
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        self.deterministic = True

        # SideNet -- conditioned on the VLM-contextualised text hidden states.
        # Its final output is the fixed-length multimodal hidden states used by
        # the injection path, already projected to the action-expert width.
        self.ft_sensor_dim = config.ft_sensor_dim
        self.num_injected_layers = config.num_injected_layers
        self.sidenet = None
        if config.num_injected_layers > 0:
            self.sidenet = nnx_bridge.ToNNX(
                SideNet(
                    branch_input_dims=(("ft_sensor", config.ft_sensor_dim),),
                    d_embedding=sidenet_cfg.d_embedding,
                    d_sidenet=sidenet_cfg.d_sidenet,
                    num_encoding_tokens=sidenet_cfg.num_encoding_tokens,
                    num_perceiver_layers=sidenet_cfg.num_perceiver_layers,
                    num_injected_layers=config.num_injected_layers,
                    num_injection_tokens=sidenet_cfg.num_injection_tokens,
                    num_heads=sidenet_cfg.num_heads,
                    vlm_hidden_dim=sidenet_cfg.vlm_hidden_dim,
                    output_hidden_features=sidenet_cfg.output_hidden_features,
                    action_expert_hidden_dim=sidenet_cfg.action_expert_hidden_dim,
                )
            )
            fake_modality_inputs = {"ft_sensor": jnp.ones((1, config.ft_sensor_dim), dtype=jnp.float32)}
            fake_text_hidden_states = jnp.ones(
                (1, config.max_token_len, sidenet_cfg.vlm_hidden_dim), dtype=jnp.float32
            )
            fake_text_mask = jnp.ones((1, config.max_token_len), dtype=jnp.bool_)
            self.sidenet.lazy_init(
                fake_modality_inputs,
                fake_text_hidden_states,
                fake_text_mask,
                deterministic=True,
                rngs=rngs,
            )

    # ------------------------------------------------------------------
    # Embedding helpers (copied verbatim from _pi0.Pi0)
    # ------------------------------------------------------------------
    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
            )
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = _pi0.posemb_sincos(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
        )
        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    # ------------------------------------------------------------------
    # SideNet plumbing
    # ------------------------------------------------------------------
    def _resolve_ft_sensor(self, observation: _model.Observation) -> jnp.ndarray:
        ft_sensor = observation.ft_sensor
        if ft_sensor is None:
            raise ValueError("Pi05WithSideNetAdapter requires `observation.ft_sensor`.")
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
            "Only single-frame ft_sensor with shape (B, 12) or (B, 1, 12) is supported."
        )

    def _compute_mm_hidden_states(
        self,
        prefix_out: jnp.ndarray,
        observation: _model.Observation,
        *,
        deterministic: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Return SideNet multimodal hidden states for injection.

        Shape: ``(B, num_injected_layers, num_injection_tokens, action_expert_width)``.
        These fixed slots are fully valid, so no mask is returned.
        """
        if self.num_injected_layers == 0:
            return None, None
        if self.sidenet is None:
            raise ValueError("SideNet is not initialized despite num_injected_layers > 0.")
        if observation.tokenized_prompt is None or observation.tokenized_prompt_mask is None:
            raise ValueError(
                "Pi05WithSideNetAdapter requires tokenized_prompt / tokenized_prompt_mask."
            )
        text_len = observation.tokenized_prompt.shape[1]
        text_features = prefix_out[:, -text_len:, :]
        text_mask = observation.tokenized_prompt_mask
        ft_sensor = self._resolve_ft_sensor(observation)
        mm_hidden_states = self.sidenet(
            {"ft_sensor": ft_sensor}, text_features, text_mask, deterministic=deterministic,
        )
        return mm_hidden_states, None

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------
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

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )

        # Pass 1: prefix-only forward. Produces the contextualised text hidden
        # states for SideNet AND the prefix KV cache consumed by Pass 2. With
        # xs[1]=None the adapter branch is dormant (inject_side=False), so the
        # adapter layers behave identically to base layers here.
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
            deterministic=not train,
        )
        mm_hidden_states, mm_hidden_states_mask = self._compute_mm_hidden_states(
            prefix_out, observation, deterministic=not train
        )

        # Pass 2: suffix forward using the prefix KV cache + SideNet injection.
        # The attention mask is (B, T_suffix, T_prefix + T_suffix).
        suffix_self_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_cross_mask = einops.repeat(
            prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
        )
        full_attn_mask = jnp.concatenate([prefix_cross_mask, suffix_self_mask], axis=-1)
        suffix_positions = (
            jnp.sum(prefix_mask, axis=-1)[:, None]
            + jnp.cumsum(suffix_mask, axis=-1)
            - 1
        )

        # At init, ``injection_gate = 0`` -> side contribution is 0 -> this
        # forward is numerically identical to base Pi0's equivalent forward,
        # so the first-step loss matches the pretrained pi0.5 loss.
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=suffix_positions,
            adarms_cond=[None, adarms_cond],
            kv_cache=kv_cache,
            fused_memory=mm_hidden_states,
            fused_memory_mask=mm_hidden_states_mask,
            deterministic=not train,
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
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

        # Prefix pass: populate KV cache AND return hidden states for SideNet.
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions,
        )
        mm_hidden_states, mm_hidden_states_mask = self._compute_mm_hidden_states(
            prefix_out, observation, deterministic=True
        )

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_step = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate([prefix_attn_mask_step, suffix_attn_mask], axis=-1)
            positions_step = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_step,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                fused_memory=mm_hidden_states,
                fused_memory_mask=mm_hidden_states_mask,
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
