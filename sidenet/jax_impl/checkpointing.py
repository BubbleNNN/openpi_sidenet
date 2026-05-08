"""Checkpoint helpers for the standalone JAX SideNet wrapper."""

from __future__ import annotations

import json
import logging
from typing import Any

from etils import epath

import openpi.shared.download as download

logger = logging.getLogger(__name__)


def maybe_get_asset_callback(model_config: Any):
    """Return a checkpoint asset saver for JAX SideNet configs when applicable."""
    from sidenet.jax_impl.pi05_with_sidenet import Pi05WithSideNetConfig
    from sidenet.jax_impl.pi05_with_sidenet_llama_adapter import Pi05WithSideNetAdapterConfig

    if not isinstance(model_config, (Pi05WithSideNetConfig, Pi05WithSideNetAdapterConfig)):
        return None

    def save_assets(directory: epath.Path) -> None:
        output_dir = directory / "sidenet_jax"
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "model_config_class": model_config.__class__.__name__,
            "sidenet_config_path": model_config.sidenet_config_path,
            "ft_sensor_dim": model_config.ft_sensor_dim,
            "pi05": model_config.pi05,
        }
        if hasattr(model_config, "num_injected_layers"):
            manifest["num_injected_layers"] = model_config.num_injected_layers
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
        )

        try:
            config_path = download.maybe_download(model_config.sidenet_config_path)
        except FileNotFoundError:
            logger.warning(
                "Could not snapshot SideNet config `%s` into checkpoint assets.",
                model_config.sidenet_config_path,
            )
            return

        (output_dir / "sidenet_config.yaml").write_text(config_path.read_text())

    return save_assets
