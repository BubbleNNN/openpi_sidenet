from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, SupportsIndex

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected in training, but keep the wrapper importable.
    torch = None


class RandomAccessDataset(Protocol):
    def __getitem__(self, index: SupportsIndex) -> Mapping[str, Any]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


def _as_scalar(value: Any) -> int:
    if torch is not None and isinstance(value, torch.Tensor):
        return int(value.item())
    return int(np.asarray(value).item())


def _zeros_like(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        return torch.zeros_like(value)
    return np.zeros_like(value)


def _copy_value(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.clone()
    return np.array(value, copy=True)


def _stack_values(values: list[Any]) -> Any:
    first = values[0]
    if torch is not None and isinstance(first, torch.Tensor):
        return torch.stack(values, dim=0)
    return np.stack(values, axis=0)


class FTWindowDatasetWrapper:
    """Wrap a random-access dataset and expose a fixed-length F/T history window.

    The wrapped dataset is expected to return samples that at least contain:
    - `ft_key`
    - `episode_key`
    - `frame_key`

    The wrapper does not mutate the underlying dataset. It returns a shallow copy
    of the current sample with an extra `output_key` field whose shape is
    `[window_size, ft_dim]`.
    """

    def __init__(
        self,
        dataset: RandomAccessDataset,
        *,
        window_size: int,
        ft_key: str = "observation.ft_sensor",
        output_key: str = "observation.ft_sensor_window",
        episode_key: str = "episode_index",
        frame_key: str = "frame_index",
        pad_mode: str = "repeat_first",
    ) -> None:
        if window_size <= 0:
            raise ValueError(f"`window_size` must be positive, got {window_size}.")
        if pad_mode not in {"repeat_first", "zeros"}:
            raise ValueError(f"Unsupported `pad_mode`: {pad_mode}")

        self._dataset = dataset
        self._window_size = window_size
        self._ft_key = ft_key
        self._output_key = output_key
        self._episode_key = episode_key
        self._frame_key = frame_key
        self._pad_mode = pad_mode

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]:
        index_int = index.__index__()
        current_sample = dict(self._dataset[index_int])
        self._validate_sample(current_sample, index_int)

        current_episode = _as_scalar(current_sample[self._episode_key])
        current_frame = _as_scalar(current_sample[self._frame_key])

        history: list[Any] = []
        cursor = index_int
        while cursor >= 0 and len(history) < self._window_size:
            candidate_sample = self._dataset[cursor]
            candidate_episode = _as_scalar(candidate_sample[self._episode_key])
            if candidate_episode != current_episode:
                break

            candidate_frame = _as_scalar(candidate_sample[self._frame_key])
            if candidate_frame > current_frame:
                raise ValueError(
                    "Dataset order is inconsistent with frame order inside the same episode. "
                    f"Current frame={current_frame}, candidate frame={candidate_frame}, index={index_int}, cursor={cursor}."
                )

            history.append(candidate_sample[self._ft_key])
            cursor -= 1

        history.reverse()
        padded_history = self._pad_history(history)
        current_sample[self._output_key] = _stack_values(padded_history)
        return current_sample

    # FIX: keep the wrapper self-contained and fail early on malformed dataset samples.
    def _validate_sample(self, sample: Mapping[str, Any], index: int) -> None:
        missing_keys = [
            key
            for key in (self._ft_key, self._episode_key, self._frame_key)
            if key not in sample
        ]
        if missing_keys:
            raise KeyError(
                f"Sample at index {index} is missing keys required by FTWindowDatasetWrapper: {missing_keys}"
            )

    # FIX: pad episode starts without crossing episode boundaries.
    def _pad_history(self, history: list[Any]) -> list[Any]:
        if not history:
            raise ValueError("F/T history is unexpectedly empty.")
        if len(history) >= self._window_size:
            return history

        pad_count = self._window_size - len(history)
        if self._pad_mode == "repeat_first":
            pad_value = history[0]
            pad_values = [_copy_value(pad_value) for _ in range(pad_count)]
        else:
            pad_value = _zeros_like(history[0])
            pad_values = [_copy_value(pad_value) for _ in range(pad_count)]
        return [*pad_values, *history]
