import dataclasses
import logging
import time

import numpy as np
from openpi_client import websocket_client_policy
import tyro

from openpi.policies import rby1_policy


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    num_steps: int = 5
    episode_length: int = 5
    prompt: str = "do something"
    zero_ft: bool = False
    ft_value: float | None = None


def make_observation(args: Args, frame_index: int) -> dict:
    obs = rby1_policy.make_rby1_example()
    obs["prompt"] = args.prompt
    obs["frame_index"] = frame_index
    obs["state"] = np.asarray(obs["state"], dtype=np.float32)
    obs["ft_sensor"] = np.asarray(obs["ft_sensor"], dtype=np.float32)

    if args.zero_ft:
        obs["ft_sensor"] = np.zeros_like(obs["ft_sensor"], dtype=np.float32)
    elif args.ft_value is not None:
        obs["ft_sensor"] = np.full_like(obs["ft_sensor"], args.ft_value, dtype=np.float32)

    return obs


def main(args: Args) -> None:
    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    logging.info("Server metadata: %s", client.get_server_metadata())

    for step in range(args.num_steps):
        frame_index = step % args.episode_length
        obs = make_observation(args, frame_index)

        start = time.time()
        result = client.infer(obs)
        elapsed_ms = 1000.0 * (time.time() - start)

        actions = np.asarray(result["actions"])
        print(
            f"step={step} frame_index={frame_index} "
            f"actions_shape={actions.shape} "
            f"mean_abs={np.abs(actions).mean():.6f} "
            f"max_abs={np.abs(actions).max():.6f} "
            f"latency_ms={elapsed_ms:.1f}"
        )

        if "policy_timing" in result:
            print("policy_timing:", result["policy_timing"])
        if "server_timing" in result:
            print("server_timing:", result["server_timing"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
