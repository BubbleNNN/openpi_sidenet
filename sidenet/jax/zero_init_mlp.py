"""Zero-initialized MLP for safe injection into pretrained models.

JAX / Flax Linen equivalent of ``sidenet.injector.zero_init_mlp``.
The last linear layer is zero-initialized so that the injection path
starts at zero, preserving the pretrained model's behaviour at the
beginning of training.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class ZeroInitMLP(nn.Module):
    """MLP with zero-initialized output projection (Flax Linen)."""

    hidden_features: int
    out_features: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_features, name="fc1")(x)
        x = nn.gelu(x)
        x = nn.Dense(
            self.out_features,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="fc2",
        )(x)
        x = nn.LayerNorm(name="norm")(x)
        return x
