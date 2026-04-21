# Server Change Log

This file records server-side changes that were reported during debugging so the
local workspace can preserve that context even when the server code is not fully
synced back.

Entries below are user-reported unless explicitly marked as mirrored locally.

## 2026-04-06 16:58:38 +0800

- Server policy direction:
  - SideNet-related training/inference should use `rby1_policy`, not `samsung_policy`.
- Server-only code presence reported:
  - Additional PyTorch model classes exist on the server, including the missing
    PI0/PI05 variants referenced by the local training code.
  - Additional config / policy modules exist on the server, including
    `pi05_ki_config` and `rby1_xhand_policy`.
- Server bug-fix status reported:
  - The `four_images` issue was already fixed on the server.
  - Other previously discussed local inconsistencies were reported as already
    fixed on the server unless noted otherwise in later entries.
- Path-handling workflow reported:
  - Hardcoded absolute paths inside `config.py` were commented out on the server
    to avoid unrelated configs touching unavailable filesystem paths during
    config loading / inspection.
  - The same strategy was mirrored locally for the custom `pi05_rby1_finetune`
    and `pi05_with_sidenet` configs.
- Local mirror status:
  - Local custom RBY1 configs now keep server-specific path fields commented out
    by default and require explicit re-enabling before use.

## 2026-04-06 17:41:12 +0800

- Server config status reported:
  - The server-side SideNet / RBY1 training path already has `use_force=False`
    for the backbone-side model config.
  - The server-side code path needed for 6-frame F/T windows and SideNet usage
    was reported as already modified by the user before the latest launch
    attempts.
- Server checkpoint situation reported:
  - The provided `pi05_base/params` path contains JAX-format checkpoint weights
    rather than a PyTorch `model.safetensors` file.

## 2026-04-06 19:09:00 +0800

- Local inference-side support mirrored for future server sync:
  - Added `Rby1FTWindowInputs` to support stateful inference-time F/T windows
    for websocket deployment, following the same pattern as
    `SamsungFTWindowInputs`.
  - Hooked `LeRobotRby1FTDataConfig` inference into `policy_config.py` so the
    official `scripts/serve_policy.py` path can materialize the F/T window
    automatically when `ft_window_size` is configured.
  - Added `scripts/serve_rby1_policy.py` as a checkpoint-only websocket server
    wrapper for RBY1 deployments. It uses the same `WebsocketPolicyServer`
    deployment form as the official OpenPI serving script.

## 2026-04-06 19:30:00 +0800

- Local inference-side validation support mirrored for future server sync:
  - Added `scripts/test_rby1_policy_server.py` as a minimal websocket client
    smoke test for RBY1 policy servers.
  - The script sends `frame_index`, single-frame `ft_sensor`, and standard
    example state/image inputs, then prints returned action shape and latency.

## Maintenance Notes

- Append new entries instead of rewriting old ones.
- Prefer recording:
  - what changed
  - whether it was user-reported or mirrored locally
  - which files / configs are affected
- If a later server change supersedes an earlier one, add a new entry rather
  than editing history in place.
