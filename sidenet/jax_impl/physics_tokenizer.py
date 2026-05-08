"""Physics-aware tokenizers for SideNet modality branches."""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp


class PhysicsTokenizer(nn.Module):
    """Tokenize raw modality vectors into fixed-width physics tokens.

    The current JAX model only wires single-frame ``ft_sensor``. For that
    branch, the 12-D vector is interpreted as four physical units:

    0. right-arm force  ``(fx, fy, fz)``
    1. right-arm torque ``(tx, ty, tz)``
    2. left-arm force   ``(fx, fy, fz)``
    3. left-arm torque  ``(tx, ty, tz)``

    Each unit is normalised, projected to ``d_embedding``, and augmented with
    learned physical-position and arm embeddings.
    """

    modality_name: str
    input_dim: int
    d_embedding: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return physics tokens with shape ``(B, L_phy, d_embedding)``."""
        if x.ndim == 3:
            if x.shape[1] != 1:
                raise ValueError(
                    "JAX PhysicsTokenizer currently supports only single-frame inputs, "
                    f"got shape {x.shape}."
                )
            x = jnp.squeeze(x, axis=1)
        if x.ndim != 2:
            raise ValueError(f"Expected input shape (B, D) or (B, 1, D), got {x.shape}.")
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.modality_name} input dim {self.input_dim}, got {x.shape[-1]}."
            )

        if self.modality_name == "ft_sensor":
            if self.input_dim != 12:
                raise ValueError(
                    f"`ft_sensor` tokenizer expects input_dim=12, got {self.input_dim}."
                )
            tokens = x.reshape(x.shape[0], 4, 3)
        else:
            # Generic fallback for low-dimensional vector modalities: each scalar
            # becomes one physical token with a learned identity embedding.
            tokens = x[..., None]

        if tokens.shape[-1] > 1:
            tokens = nn.LayerNorm(name="physical_unit_norm")(tokens)
        tokens = nn.Dense(self.d_embedding, name="token_fc1")(tokens)
        tokens = nn.gelu(tokens)
        tokens = nn.Dense(self.d_embedding, name="token_fc2")(tokens)

        physical_embedding = self.param(
            "physical_embedding",
            nn.initializers.normal(stddev=0.02),
            (tokens.shape[1], self.d_embedding),
        )
        tokens = tokens + physical_embedding[None, :, :].astype(tokens.dtype)

        if self.modality_name == "ft_sensor":
            arm_embedding = self.param(
                "arm_embedding",
                nn.initializers.normal(stddev=0.02),
                (2, self.d_embedding),
            )
            arm_indices = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
            tokens = tokens + arm_embedding[arm_indices][None, :, :].astype(tokens.dtype)

        return tokens
