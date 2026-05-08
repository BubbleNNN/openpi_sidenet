"""Custom Gemma with split-softmax multimodal injection in the last N layers.

This file provides a drop-in replacement for ``openpi.models.gemma.Module``
whose **last** ``num_injected_layers`` transformer blocks replace the base
``Attention`` with split-softmax injection attention.

    - Injection tokens are inserted at the *last* L Transformer layers.
    - SideNet produces action-width multimodal hidden states with shape
      ``(B, L, K, D_expert)``.
    - The scanned injected layer ``l`` directly receives
      ``mm_hidden_states[:, l]`` as its side tokens.

Per-layer injection mechanics
-----------------------------
For each of the last ``num_injected_layers`` blocks, the side tokens are::

    side_l = mm_hidden_states[:, l, :, :]         # (B, K, D_expert)

Then ``side_l`` is projected with the action-expert Q/K/V weights and
prepended before the original K/V sequence. The side rows get position 0
(RoPE identity), the QK logits are split into a side block and an orig
block, and the side block is multiplied by a per-layer, per-head
``tanh(injection_gate)`` (zero-init) and a per-query mask that zeroes out
queries outside the injected expert stream.

Identity at initialisation
--------------------------
``injection_gate = 0`` → ``tanh(0) = 0`` → side-attention contribution is
exactly zero at step 0. SideNet's normally initialised hidden states are
non-zero from the start, giving the gate a non-zero gradient. When the base
pi0.5 weights are loaded into the non-injection parameters, the first
forward/backward produces the pretrained pi0.5 loss.

Weight layout
-------------
Base ``gemma.Module`` stores all ``depth`` transformer blocks in a
single ``nn.scan``-packed parameter tree under ``layers/...`` with a
leading axis of length ``depth``. This module instead uses two
consecutive scans:

    base_layers/     ... scan length = depth - num_injected_layers
    injected_layers/ ... scan length = num_injected_layers

Base pretrained weights are split along axis 0 so that the first
``depth - num_injected_layers`` slices go into ``base_layers`` and the
last ``num_injected_layers`` slices go into ``injected_layers``. See
``sidenet.jax_impl.weight_loader.split_scanned_layers_for_adapter``.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models import gemma as _gemma
from openpi.models import lora
import openpi.training.sharding as sharding

# Re-export so callers don't have to reach into openpi.models.gemma.
Config = _gemma.Config
RMSNorm = _gemma.RMSNorm
Embedder = _gemma.Embedder
FeedForward = _gemma.FeedForward
KVCache = _gemma.KVCache
_apply_rope = _gemma._apply_rope
_name = _gemma._name
_gated_residual = _gemma._gated_residual
PALIGEMMA_VOCAB_SIZE = _gemma.PALIGEMMA_VOCAB_SIZE

logger = logging.getLogger(__name__)


class InjectionAttention(nn.Module):
    """Base Gemma attention + split-softmax injection on ``injection_expert_idx``.

    At initialisation (``injection_gate = 0``), this produces exactly the
    same output as ``gemma.Attention`` over the same ``xs``. The only new
    trainable parameter in this attention block is ``injection_gate``
    ``(num_heads,)``, zero-initialised per scanned layer. Every Q/K/V/O
    einsum is named identically to base ``gemma.Attention`` so weights load 1:1.
    """

    configs: Sequence[Config]
    injection_expert_idx: int = 1
    num_injection_tokens: int = 8

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache, injection_hidden_states):
        # xs: list of per-expert token streams, or None.
        # injection_hidden_states: (B, K, D_expert), the current scanned layer's
        # SideNet multimodal hidden states, or None.
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        num_heads = self.configs[0].num_heads
        num_kv_heads = self.configs[0].num_kv_heads
        head_dim = self.configs[0].head_dim
        dtype = next(x.dtype for x in xs if x is not None)

        # Build the per-expert input for Q/K/V projection. The side tokens are
        # temporarily inserted into the injected expert so they reuse that
        # expert's base Q/K/V weights. After projection, they are moved to the
        # front of the full key/value sequence to implement K/V = [M, P].
        inject_side = (
            injection_hidden_states is not None and xs[self.injection_expert_idx] is not None
        )

        if inject_side:
            injection_cfg = self.configs[self.injection_expert_idx]
            if injection_hidden_states.ndim != 3:
                raise ValueError(
                    "injection_hidden_states must have shape (B, K, D_expert), "
                    f"got {injection_hidden_states.shape}."
                )
            if injection_hidden_states.shape[1] != self.num_injection_tokens:
                raise ValueError(
                    f"injection_hidden_states token count {injection_hidden_states.shape[1]} "
                    f"must equal num_injection_tokens {self.num_injection_tokens}."
                )
            if injection_hidden_states.shape[-1] != injection_cfg.width:
                raise ValueError(
                    f"injection_hidden_states last dim {injection_hidden_states.shape[-1]} "
                    f"must equal injected expert width {injection_cfg.width}."
                )
            side_tokens = injection_hidden_states.astype(dtype)
        else:
            side_tokens = None

        k_side = side_tokens.shape[1] if side_tokens is not None else 0
        xs_proj = list(xs)
        if side_tokens is not None:
            xs_proj[self.injection_expert_idx] = jnp.concatenate(
                [side_tokens, xs[self.injection_expert_idx]], axis=1
            )

        # Per-expert QKV projection (names and init match base gemma.Attention so
        # pretrained weights load 1:1).
        qkvs = []
        for i, (x, config) in enumerate(zip(xs_proj, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q_aug, k_aug, v_aug = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        if inject_side:
            # ``q_aug/k_aug/v_aug`` currently have sequence order
            # [tokens_before_injected_expert, side, injected_expert_tokens,
            # tokens_after_injected_expert]. Move side to the global front so
            # the full K/V order becomes [side, original_tokens]. This matches
            # the intended split-attention formula K,V = [M, P].
            side_start_current = sum(
                xs[i].shape[1] for i in range(self.injection_expert_idx) if xs[i] is not None
            )
            side_end_current = side_start_current + k_side

            def move_side_to_front(y):
                y_side = y[:, side_start_current:side_end_current]
                y_orig = jnp.concatenate(
                    [y[:, :side_start_current], y[:, side_end_current:]], axis=1
                )
                return jnp.concatenate([y_side, y_orig], axis=1)

            q_aug = move_side_to_front(q_aug)
            k_aug = move_side_to_front(k_aug)
            v_aug = move_side_to_front(v_aug)

            # Side positions are set to 0 so RoPE becomes identity (cos(0)=1,
            # sin(0)=0). Original token positions remain in their base order.
            pos_side = jnp.zeros((positions.shape[0], k_side), dtype=positions.dtype)
            positions_aug = jnp.concatenate([pos_side, positions], axis=1)
        else:
            positions_aug = positions

        q_aug = _apply_rope(q_aug, positions=positions_aug)
        q_aug = q_aug * (head_dim ** -0.5)
        k_aug = _apply_rope(k_aug, positions=positions_aug)

        if inject_side:
            # Q should contain only original query rows. K/V keep side rows at
            # the global front, followed by the original sequence. If a prefix
            # cache exists, it is part of the original sequence P, so K/V order
            # is [side, cache, current_original].
            q = q_aug[:, k_side:]
            k_side_rows = k_aug[:, :k_side]
            k_current_orig = k_aug[:, k_side:]
            v_side_rows = v_aug[:, :k_side]
            v_current_orig = v_aug[:, k_side:]
            if kv_cache is not None:
                cache_k, cache_v = kv_cache
                k = jnp.concatenate([k_side_rows, cache_k, k_current_orig], axis=1)
                v = jnp.concatenate([v_side_rows, cache_v, v_current_orig], axis=1)
                cache_len = cache_k.shape[1]
            else:
                k = k_aug
                v = v_aug
                cache_len = 0
        else:
            # Base path: identical to gemma.Attention.
            if kv_cache is not None:
                cache_k, cache_v = kv_cache
                k_aug = jnp.concatenate([cache_k, k_aug], axis=1)
                v_aug = jnp.concatenate([cache_v, v_aug], axis=1)
            else:
                cache_len = 0
            q = q_aug
            k = k_aug
            v = v_aug

        q5 = einops.rearrange(q, "B T (K G) H -> B T K G H", K=num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q5, k, preferred_element_type=jnp.float32)

        # Expected attention mask: (B, 1, T_q_orig, T_k_orig) with both axes
        # excluding side rows. T_k_orig *does* include any cache rows.
        b = q.shape[0]
        t_q = q.shape[1]
        t_k_orig = k.shape[1] - k_side
        if attn_mask.shape != (b, 1, t_q, t_k_orig):
            raise ValueError(
                f"Attention mask shape {attn_mask.shape} is incompatible with q/k shapes "
                f"{q.shape}/{k.shape} after dropping {k_side} side rows."
            )
        big_neg = -2.3819763e38

        if inject_side:
            # K/V are ordered as [side, original], so the key-axis split is direct.
            logits_side = logits[..., :k_side]
            logits_orig = logits[..., k_side:]
            mask_orig = attn_mask[:, :, None, :, :]  # (B, 1, 1, T_q, T_k_orig)
            masked_orig = jnp.where(mask_orig, logits_orig, big_neg)
            probs_orig = jax.nn.softmax(masked_orig, axis=-1).astype(dtype)
            probs_side = jax.nn.softmax(logits_side.astype(jnp.float32), axis=-1).astype(dtype)

            # Per-head tanh gate (zero-init -> identity to base at step 0).
            gate = self.param("injection_gate", nn.initializers.zeros_init(), (num_heads,))
            gate = jnp.tanh(gate).astype(dtype)
            gate = einops.rearrange(gate, "(K G) -> K G", K=num_kv_heads)

            # Build a per-query mask that is 1 only for queries from the injected
            # expert (side injection must not leak into other experts).
            query_is_injection_expert = jnp.zeros((t_q,), dtype=dtype)
            cursor = 0
            for i, (x, _cfg) in enumerate(zip(xs, self.configs, strict=True)):
                if x is None:
                    continue
                tlen = x.shape[1]
                if i == self.injection_expert_idx:
                    query_is_injection_expert = query_is_injection_expert.at[
                        cursor : cursor + tlen
                    ].set(1.0)
                cursor += tlen

            probs_side = probs_side * gate[None, :, :, None, None]
            probs_side = probs_side * query_is_injection_expert[None, None, None, :, None]

            # Restore into full-key layout matching v: [side, original].
            probs_full = jnp.concatenate([probs_side, probs_orig], axis=-1)
        else:
            masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)
            probs_full = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs_full, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        # Per-expert output projection over ORIGINAL (non-side) query positions.
        # encoded has shape (B, T_q, N*H) where T_q = sum of orig lengths across experts.
        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                out.append(None)
                continue
            end = start + x.shape[1]
            out_einsum = lora.Einsum(
                shape=(config.num_heads, config.head_dim, config.width),
                name=_name("attn_vec_einsum", i),
                init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                lora_config=config.lora_configs.get("attn"),
            )
            out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
            start = end

        # Return the original (non-side) K/V. Shape matches base gemma's
        # returned cache exactly (including any prepended cache rows), so that
        # downstream callers that carry (k, v) across calls see the same layout.
        if inject_side:
            k_orig = k[:, k_side:]
            v_orig = v[:, k_side:]
        else:
            k_orig, v_orig = k, v
        return out, (k_orig, v_orig)


class InjectionBlock(nn.Module):
    """Transformer block identical to ``gemma.Block`` but using ``InjectionAttention``.

    Accepts the current layer's ``injection_hidden_states`` slice from SideNet.
    """

    configs: tuple[Config, ...]
    injection_expert_idx: int = 1
    num_injection_tokens: int = 8
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(  # noqa: FBT002
        self,
        xs,
        kv_cache,
        positions,
        attn_mask,
        adarms_cond,
        injection_hidden_states,
        deterministic=True,
    ):
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        attn = InjectionAttention(
            configs=self.configs,
            injection_expert_idx=self.injection_expert_idx,
            num_injection_tokens=self.num_injection_tokens,
            name="attn",
        )

        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            gate = None
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
            pre_attn.append(x)
            gates.append(gate)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache, injection_hidden_states)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            gate = None
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
                x = lora.FeedForward(  # noqa: PLW2901
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)
            out.append(x)
            gates.append(gate)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache


class InjectionModule(nn.Module):
    """Drop-in replacement for ``gemma.Module`` with final injected blocks."""

    configs: Sequence[Config]
    embed_dtype: str
    num_injected_layers: int = 5
    injection_expert_idx: int = 1
    num_injection_tokens: int = 8
    fused_memory_dim: int = 1024  # SideNet final output / action-expert width

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    adarms: bool = False

    def setup(self):
        assert all(config.depth == self.configs[0].depth for config in self.configs)
        assert 0 <= self.num_injected_layers < self.configs[0].depth
        if self.num_injected_layers == 0:
            logger.warning(
                "InjectionModule initialized with num_injected_layers=0; "
                "SideNet injection is disabled."
            )
        else:
            logger.info(
                "InjectionModule initialized with num_injected_layers=%d "
                "(injecting the last %d transformer layers).",
                self.num_injected_layers,
                self.num_injected_layers,
            )

        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,
            name="embedder",
        )

        # Base layers run FIRST, injected layers run LAST.
        # SideNet provides one K-token hidden-state slice for each injected layer.
        base_block_cls = nn.remat(
            _gemma.Block,
            prevent_cse=False,
            static_argnums=(5,),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.base_layers = nn.scan(
            base_block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.configs[0].depth - self.num_injected_layers,
        )(
            configs=tuple(self.configs),
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )

        if self.num_injected_layers > 0:
            injection_block_cls = nn.remat(
                InjectionBlock,
                prevent_cse=False,
                static_argnums=(6,),  # deterministic (arg 6 after self)
                policy=jax.checkpoint_policies.nothing_saveable,
            )
            self.injected_layers = nn.scan(
                injection_block_cls,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=(
                    0,              # kv_cache (axis 0 is layer)
                    nn.broadcast,   # positions
                    nn.broadcast,   # attn_mask
                    nn.broadcast,   # adarms_cond
                    0,              # injection_hidden_states (axis 0 is layer)
                    nn.broadcast,   # deterministic
                ),
                length=self.num_injected_layers,
            )(
                configs=tuple(self.configs),
                injection_expert_idx=self.injection_expert_idx,
                num_injection_tokens=self.num_injection_tokens,
                dropout=self.dropout,
                dropout_bdims=self.dropout_bdims,
            )

        self.final_norms = [
            RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))
        ]

    def embed(self, tokens):
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    def __call__(
        self,
        embedded,
        positions,
        mask,
        adarms_cond=None,
        *,
        fused_memory=None,
        fused_memory_mask=None,
        kv_cache=None,
        deterministic: bool = True,
    ):
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        first_stream = next(e for e in embedded if e is not None)
        batch_size = first_stream.shape[0]
        dtype = first_stream.dtype

        if self.num_injected_layers == 0:
            if fused_memory is not None and fused_memory.shape[1] != 0:
                raise ValueError(
                    "fused_memory must be None or have zero injected-layer length "
                    "when num_injected_layers=0."
                )
            embedded, kv_cache_out = self.base_layers(
                embedded, kv_cache, positions, mask, adarms_cond, deterministic
            )
            assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)
            return [
                f(e, a)[0] if e is not None else e
                for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
            ], kv_cache_out

        injection_width = self.configs[self.injection_expert_idx].width

        # SideNet final output is the action-width multimodal hidden-state table
        # used for injection: (B, L, K, D_expert). Transpose to scan layout
        # (L, B, K, D_expert) so every injected block receives its own slice.
        if fused_memory is not None:
            if fused_memory.ndim != 4:
                raise ValueError(
                    "fused_memory must have shape (B, L, K, D_expert), "
                    f"got {fused_memory.shape}."
                )
            expected_shape = (
                batch_size,
                self.num_injected_layers,
                self.num_injection_tokens,
                injection_width,
            )
            if fused_memory.shape != expected_shape:
                raise ValueError(
                    f"fused_memory shape {fused_memory.shape} must equal {expected_shape} "
                    "for layer-wise split-attention injection."
                )
            if fused_memory_mask is not None:
                if fused_memory_mask.shape != fused_memory.shape[:3]:
                    raise ValueError(
                        f"fused_memory_mask shape {fused_memory_mask.shape} does not "
                        f"match fused_memory shape {fused_memory.shape[:3]}."
                    )
                fused_memory = jnp.where(
                    fused_memory_mask[..., None],
                    fused_memory,
                    jnp.zeros_like(fused_memory),
                )
            injection_hidden_states = jnp.swapaxes(fused_memory.astype(dtype), 0, 1)
        else:
            if self.fused_memory_dim != injection_width:
                raise ValueError(
                    f"fused_memory_dim {self.fused_memory_dim} must match injected "
                    f"expert width {injection_width}. The SideNet output MLP should "
                    "project to the base model injection width."
                )
            injection_hidden_states = jnp.zeros(
                (
                    self.num_injected_layers,
                    batch_size,
                    self.num_injection_tokens,
                    injection_width,
                ),
                dtype=dtype,
            )

        # Split incoming kv_cache between early base layers and final injected
        # layers along the original layer axis.
        base_length = self.configs[0].depth - self.num_injected_layers
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            base_cache = (cache_k[:base_length], cache_v[:base_length])
            injected_cache = (cache_k[base_length:], cache_v[base_length:])
        else:
            base_cache = None
            injected_cache = None

        embedded, base_cache_out = self.base_layers(
            embedded, base_cache, positions, mask, adarms_cond, deterministic
        )
        embedded, injected_cache_out = self.injected_layers(
            embedded,
            injected_cache,
            positions,
            mask,
            adarms_cond,
            injection_hidden_states,
            deterministic,
        )

        # Re-stack layer caches along axis 0 so downstream code (pi0 sample_actions)
        # that treats kv_cache as a single (k, v) tuple with leading layer axis of
        # size ``depth`` still works unchanged.
        if base_cache_out is not None and injected_cache_out is not None:
            k_out = jnp.concatenate([base_cache_out[0], injected_cache_out[0]], axis=0)
            v_out = jnp.concatenate([base_cache_out[1], injected_cache_out[1]], axis=0)
            kv_cache_out = (k_out, v_out)
        else:
            kv_cache_out = (base_cache_out, injected_cache_out)

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)
        return [
            f(e, a)[0] if e is not None else e
            for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache_out

    def init(self, use_adarms: Sequence[bool]):
        """Linen-init helper, mirrors ``gemma.Module.init`` plus a fake fused_memory."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        # A fake fused_memory with SideNet's final output / action-expert width.
        fused_memory = None
        if self.num_injected_layers > 0:
            fused_memory = jnp.zeros(
                (1, self.num_injected_layers, self.num_injection_tokens, self.fused_memory_dim)
            )
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[
                jnp.zeros((1, c.width)) if u else None
                for u, c in zip(use_adarms, self.configs, strict=True)
            ],
            fused_memory=fused_memory,
        )
