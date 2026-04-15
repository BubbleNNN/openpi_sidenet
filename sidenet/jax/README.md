# JAX SideNet

This directory contains the JAX SideNet path that stays local to `sidenet/jax`.

The key point is that the integration is done by a standalone wrapper model,
not by modifying `src/openpi/models/pi0.py`.

## Main entry

- `pi05_with_sidenet.py`
  - `Pi05WithSideNetConfig`
  - `Pi05WithSideNet`

This wrapper composes the existing JAX `Pi0/Pi0.5` model with `sidenet/jax/SideNet`
and injects SideNet output tokens into the suffix/action-expert path.

## Current behavior

- Only `pi05=True` is supported.
- Only single-frame `ft_sensor` is supported.
- Accepted `ft_sensor` shapes are `(B, 12)` and `(B, 1, 12)`.
- Multi-frame F/T windows are intentionally rejected.
- Text conditioning comes from contextualized VLM prefix hidden states, not raw token embeddings.
- Injection is done by prepending SideNet tokens to the suffix sequence before the action expert.

## Config names

Two training configs are wired into `src/openpi/training/config.py`:

- `debug_pi05_with_sidenet_jax_smoke`
- `pi05_with_sidenet_jax`

The old `pi05_with_sidenet` config is still the PyTorch SideNet path.

## Config files

- Production config: `sidenet/sidenet_config.yaml`
- Smoke config: `sidenet/sidenet_smoke_config.yaml`

Both are now aligned to the current v2 perceiver-based schema and only define
the `ft_sensor` branch with `input_dim: 12`.

## Training

If you need fresh normalization statistics:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_with_sidenet_jax
```

Smoke run:

```bash
uv run scripts/train.py debug_pi05_with_sidenet_jax_smoke --exp-name=smoke --overwrite
```

Full run:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_with_sidenet_jax --exp-name=my_experiment --overwrite
```

## Inference

Once training has produced a JAX checkpoint, use the same config name with the
standard policy server:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_with_sidenet_jax \
    --policy.dir=checkpoints/pi05_with_sidenet_jax/my_experiment/20000
```

## Notes

- The wrapper is loaded through the normal JAX `TrainConfig.model.create/load` path.
- `sidenet.enabled` is left `False` for the JAX config so inference does not get
  misrouted into the PyTorch split-SideNet loading branch.
- If you want to change the SideNet architecture, update the YAML file and keep
  the branch name as `ft_sensor` unless you also update the wrapper.
