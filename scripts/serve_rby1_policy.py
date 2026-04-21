import dataclasses
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    # Training config name.
    config: str = "pi05_with_sidenet"
    # Checkpoint directory, e.g. /path/to/checkpoints/pi05_with_sidenet/train/20000
    dir: str = tyro.MISSING
    # Default prompt when the request does not contain one.
    default_prompt: str | None = None
    # Port to serve on.
    port: int = 8000
    # Record requests and responses for debugging.
    record: bool = False
    # Optional explicit PyTorch device, e.g. cuda:0
    pytorch_device: str | None = None


def main(args: Args) -> None:
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(
        train_config,
        args.dir,
        default_prompt=args.default_prompt,
        pytorch_device=args.pytorch_device,
    )
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating RBY1 server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
