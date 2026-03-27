import dataclasses
from collections import deque

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_samsung_example() -> dict:
    """Creates a random input example for the Samsung policy."""
    return {
        # FIX: Samsung raw policy inputs use the same dot-delimited keys as the
        # LeRobot-style dataset contract so training and inference stay aligned.
        "observation.images.cam_high_left": np.random.randint(256, size=(3, 500, 800), dtype=np.uint8),
        "observation.images.cam_high_right": np.random.randint(256, size=(3, 500, 800), dtype=np.uint8),
        "observation.images.cam_left_wrist": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "observation.images.cam_right_wrist": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "observation.state": np.random.rand(22).astype(np.float32),
        "observation.ft_sensor": np.random.rand(12).astype(np.float32),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SamsungInputs(transforms.DataTransformFn):
    """Inputs for the Samsung RBY1 policy.

    Expected inputs:
    - observation.images.cam_high_left: [channel, height, width]
    - observation.images.cam_high_right: [channel, height, width]
    - observation.images.cam_left_wrist: [channel, height, width]
    - observation.images.cam_right_wrist: [channel, height, width]
    - observation.state: [22]
    - observation.ft_sensor: [12]
    - actions: [action_horizon, 22] (optional, training only)
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # FIX: Samsung policy expects dot-delimited raw keys so inference uses
        # the same field names as the Samsung training data contract.
        state = np.asarray(data["observation.state"], dtype=np.float32)
        # FIX: when an F/T window has been materialized upstream, use it
        # directly; otherwise fall back to the single-frame sensor reading.
        ft_key = "observation.ft_sensor_window" if "observation.ft_sensor_window" in data else "observation.ft_sensor"
        force_torque = np.asarray(data[ft_key], dtype=np.float32)

        # LeRobot video frames may come in float32 CHW. Convert to uint8 HWC.
        high_left_image = _parse_image(data["observation.images.cam_high_left"])
        high_right_image = _parse_image(data["observation.images.cam_high_right"])
        left_wrist_image = _parse_image(data["observation.images.cam_left_wrist"])
        right_wrist_image = _parse_image(data["observation.images.cam_right_wrist"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                # Current pi0/pi05 image contract exposes a single base camera slot.
                images = {
                    "base_0_rgb": high_left_image,
                    "left_wrist_0_rgb": left_wrist_image,
                    "right_wrist_0_rgb": right_wrist_image,
                }
                image_masks = {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.True_,
                    "right_wrist_0_rgb": np.True_,
                }
            case _model.ModelType.PI0_FAST:
                images = {
                    "base_0_rgb": high_left_image,
                    "base_1_rgb": high_right_image,
                    "wrist_0_rgb": left_wrist_image,
                }
                image_masks = {
                    "base_0_rgb": np.True_,
                    "base_1_rgb": np.True_,
                    "wrist_0_rgb": np.True_,
                }
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
            "force_torque": force_torque,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SamsungFTWindowInputs(SamsungInputs):
    """Stateful Samsung inference inputs that build an F/T history window.

    This transform is intended for inference only. Training should keep using
    the dataset-side `FTWindowDatasetWrapper`.
    """

    window_size: int = 1
    pad_mode: str = "repeat_first"
    _ft_history: deque = dataclasses.field(init=False, repr=False, compare=False)
    _last_frame_index: int | None = dataclasses.field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError(f"`window_size` must be positive, got {self.window_size}.")
        if self.pad_mode not in {"repeat_first", "zeros"}:
            raise ValueError(f"Unsupported `pad_mode`: {self.pad_mode}")
        object.__setattr__(self, "_ft_history", deque(maxlen=self.window_size))
        object.__setattr__(self, "_last_frame_index", None)

    # FIX: infer-time F/T windows should be built inside the Samsung-specific
    # transform layer, not inside the generic Policy or server code.
    def __call__(self, data: dict) -> dict:
        if "observation.ft_sensor_window" not in data:
            data = dict(data)
            current_ft = np.asarray(data["observation.ft_sensor"], dtype=np.float32)

            frame_index = None
            if "frame_index" in data:
                frame_index = int(np.asarray(data["frame_index"]).item())
                if frame_index == 0 or (
                    self._last_frame_index is not None and frame_index <= self._last_frame_index
                ):
                    self._ft_history.clear()
                object.__setattr__(self, "_last_frame_index", frame_index)

            self._ft_history.append(current_ft)
            history = list(self._ft_history)
            if len(history) < self.window_size:
                pad_count = self.window_size - len(history)
                if self.pad_mode == "repeat_first":
                    pad_value = np.array(history[0], copy=True)
                else:
                    pad_value = np.zeros_like(history[0])
                history = [np.array(pad_value, copy=True) for _ in range(pad_count)] + history

            data["observation.ft_sensor_window"] = np.stack(history, axis=0)

        return super().__call__(data)


@dataclasses.dataclass(frozen=True)
class SamsungOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :22])}
