import dataclasses
from collections import deque
from typing import ClassVar
import einops
import numpy as np
from openpi import transforms
RBY1_ACTION_DIM = 22

def make_rby1_example() -> dict:
    
    return {
        "state": np.ones((RBY1_ACTION_DIM,)),
        "ft_sensor": np.ones((12,)),
        "images":{
            "cam_high_left":np.random.randint(256, size=(3,224,224), dtype=np.uint8),
            "cam_high_right":np.random.randint(256, size=(3,224,224), dtype=np.uint8),
            "cam_left_wrist":np.random.randint(256, size=(3,224,224), dtype=np.uint8),
            "cam_right_wrist":np.random.randint(256, size=(3,224,224), dtype=np.uint8),
        },
        "prompt" :"do something"
    }
    
@dataclasses.dataclass(frozen=True)
class Rby1Inputs(transforms.DataTransformFn):
    action_dim : int
    exclude_torso: bool=False
    use_cam_high_right: bool=False
    
    exclude_gripper_from_state: bool=False
    
    EXPECTED_CAMERAS: ClassVar[tuple[str,...]] = {
        "cam_high_left",
        "cam_high_right",
        "cam_left_wrist",
        "cam_right_wrist",
    }
    
    def __call__(self, data:dict)->dict:
        data = _decode_rby1(data)
        start_idx = 6 if self.exclude_torso else 0
        valid_action_dim = RBY1_ACTION_DIM - start_idx
        
        state_slice = data["state"][start_idx:-2] if self.exclude_gripper_from_state else data["state"][start_idx:]
        state = transforms.pad_to_dim(state_slice, self.action_dim)
        
        in_images = data["images"]
        
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Rby1Inputs expects {self.EXPECTED_CAMERAS} cameras, but got {set(in_images)}")
        
        base_image = in_images['cam_high_right'] if self.use_cam_high_right else in_images['cam_high_left']
        
        images = {
            "base_0_rgb" : base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }
        extra_image_names = {
            "left_wrist_0_rgb":"cam_left_wrist",
            "right_wrist_0_rgb":"cam_right_wrist",
        }        
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_
        
        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }
        
        if "ft_sensor" in data:
            inputs["ft_sensor"] = np.asarray(data["ft_sensor"])
            
        if "actions" in data:
            actions = np.asarray(data["actions"][:,start_idx:])
            assert actions.shape[1] == valid_action_dim
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)
        
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        
        return inputs
    
@dataclasses.dataclass(frozen=True)
class Rby1Outputs(transforms.DataTransformFn):
    exclude_torso: bool=False
    
    def __call__(self, data : dict) -> dict:
        valid_action_dim = RBY1_ACTION_DIM - (6 if self.exclude_torso else 0)
        actions = np.asarray(data["actions"])[:,: valid_action_dim]
        return {"actions": actions}


@dataclasses.dataclass(frozen=True)
class Rby1FTWindowInputs(Rby1Inputs):
    """Stateful RBY1 inference inputs that build an F/T history window.

    This mirrors the dataset-side `FTWindowDatasetWrapper` during deployment so
    the websocket policy server can keep using the standard official
    `serve_policy.py` path.
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

    def __call__(self, data: dict) -> dict:
        data = dict(data)

        if "ft_sensor_window" in data:
            data["ft_sensor"] = np.asarray(data["ft_sensor_window"], dtype=np.float32)
            return super().__call__(data)

        if "ft_sensor" not in data:
            return super().__call__(data)

        current_ft = np.asarray(data["ft_sensor"], dtype=np.float32)

        frame_index = None
        if "frame_index" in data:
            frame_index = int(np.asarray(data["frame_index"]).item())
            if frame_index == 0 or (self._last_frame_index is not None and frame_index <= self._last_frame_index):
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

        data["ft_sensor_window"] = np.stack(history, axis=0)
        data["ft_sensor"] = data["ft_sensor_window"]
        return super().__call__(data)
    
def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val

def _decode_rby1(data: dict) -> dict:
    state = np.asarray(data["state"])
    def convert_image(img):
        img = np.asarray(img)
        if np.issubdtype(img.dtype, np.floating):
            img = (255*img).astype(np.uint8)
        
        return einops.rearrange(img, "c h w->h w c")
    
    images = data['images']
    images_dict = {name: convert_image(img) for name, img in images.items()}
    
    data["images"] = images_dict
    data["state"] = state
    return data
