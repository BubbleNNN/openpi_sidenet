#!/usr/bin/env bash

set -euo pipefail

# Configuration
# Fill these in with your real training commands.
# The script will prepend CUDA_VISIBLE_DEVICES automatically.
SIDENET_PI05_CMD='__FILL_ME__'
PI05_FINETUNE_CMD='__FILL_ME__'

# Matching policy:
# - exact: launch only when free GPU count is exactly 4 or exactly 8
# - at_least: launch when free GPU count is >= 4 or >= 8
MATCH_MODE="${MATCH_MODE:-exact}"

# Polling interval in seconds while waiting for a matching GPU count.
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-60}"

# A GPU is considered free only if all three conditions hold:
# 1. No active compute process is attached to it.
# 2. Memory used is below GPU_MEM_USED_THRESHOLD_MB.
# 3. Utilization is below GPU_UTIL_THRESHOLD_PERCENT.
GPU_MEM_USED_THRESHOLD_MB="${GPU_MEM_USED_THRESHOLD_MB:-1024}"
GPU_UTIL_THRESHOLD_PERCENT="${GPU_UTIL_THRESHOLD_PERCENT:-10}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

validate_configuration() {
  if [[ "$SIDENET_PI05_CMD" == "__FILL_ME__" ]]; then
    echo "Set SIDENET_PI05_CMD at the top of this script before running it." >&2
    exit 1
  fi

  if [[ "$PI05_FINETUNE_CMD" == "__FILL_ME__" ]]; then
    echo "Set PI05_FINETUNE_CMD at the top of this script before running it." >&2
    exit 1
  fi

  if [[ "$MATCH_MODE" != "exact" && "$MATCH_MODE" != "at_least" ]]; then
    echo "MATCH_MODE must be 'exact' or 'at_least', got: $MATCH_MODE" >&2
    exit 1
  fi
}

gpu_count_matches() {
  local free_count="$1"
  local target="$2"

  if [[ "$MATCH_MODE" == "exact" ]]; then
    [[ "$free_count" -eq "$target" ]]
  else
    [[ "$free_count" -ge "$target" ]]
  fi
}

get_free_gpu_ids() {
  local -A busy_gpu_uuids=()

  while IFS=',' read -r gpu_uuid pid; do
    gpu_uuid="${gpu_uuid// /}"
    pid="${pid// /}"
    if [[ -n "$gpu_uuid" && -n "$pid" ]]; then
      busy_gpu_uuids["$gpu_uuid"]=1
    fi
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)

  while IFS=',' read -r index uuid memory_used gpu_util; do
    index="${index// /}"
    uuid="${uuid// /}"
    memory_used="${memory_used// /}"
    gpu_util="${gpu_util// /}"

    if [[ -z "$index" || -z "$uuid" || -z "$memory_used" || -z "$gpu_util" ]]; then
      continue
    fi

    if [[ -n "${busy_gpu_uuids[$uuid]:-}" ]]; then
      continue
    fi

    if (( memory_used > GPU_MEM_USED_THRESHOLD_MB )); then
      continue
    fi

    if (( gpu_util > GPU_UTIL_THRESHOLD_PERCENT )); then
      continue
    fi

    echo "$index"
  done < <(nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
}

join_by_comma() {
  local IFS=','
  echo "$*"
}

launch_command() {
  local gpu_ids_csv="$1"
  local command_string="$2"
  local tag="$3"

  echo "[$(timestamp)] Launching ${tag} on GPUs: ${gpu_ids_csv}" >&2
  CUDA_VISIBLE_DEVICES="$gpu_ids_csv" bash -lc "$command_string" &
  echo "$!"
}

main() {
  require_command nvidia-smi
  validate_configuration

  log "Waiting for free GPUs. MATCH_MODE=${MATCH_MODE}, MEM<=${GPU_MEM_USED_THRESHOLD_MB}MB, UTIL<=${GPU_UTIL_THRESHOLD_PERCENT}%"

  while true; do
    mapfile -t free_gpu_ids < <(get_free_gpu_ids)
    free_gpu_count="${#free_gpu_ids[@]}"

    if (( free_gpu_count == 0 )); then
      log "No free GPUs detected."
      sleep "$CHECK_INTERVAL_SEC"
      continue
    fi

    log "Free GPUs detected (${free_gpu_count}): $(join_by_comma "${free_gpu_ids[@]}")"

    if gpu_count_matches "$free_gpu_count" 8; then
      sidenet_gpu_ids=("${free_gpu_ids[@]:0:4}")
      finetune_gpu_ids=("${free_gpu_ids[@]:4:4}")

      if (( ${#sidenet_gpu_ids[@]} < 4 || ${#finetune_gpu_ids[@]} < 4 )); then
        log "Matched 8-GPU policy, but failed to allocate two 4-GPU groups. Retrying."
        sleep "$CHECK_INTERVAL_SEC"
        continue
      fi

      sidenet_pid="$(launch_command "$(join_by_comma "${sidenet_gpu_ids[@]}")" "$SIDENET_PI05_CMD" "SideNet+PI05 training")"
      finetune_pid="$(launch_command "$(join_by_comma "${finetune_gpu_ids[@]}")" "$PI05_FINETUNE_CMD" "PI05 finetuning")"

      log "Started SideNet+PI05 training pid=${sidenet_pid}"
      log "Started PI05 finetuning pid=${finetune_pid}"
      wait "$sidenet_pid" "$finetune_pid"
      exit $?
    fi

    if gpu_count_matches "$free_gpu_count" 4; then
      sidenet_gpu_ids=("${free_gpu_ids[@]:0:4}")

      if (( ${#sidenet_gpu_ids[@]} < 4 )); then
        log "Matched 4-GPU policy, but failed to allocate 4 GPUs. Retrying."
        sleep "$CHECK_INTERVAL_SEC"
        continue
      fi

      sidenet_pid="$(launch_command "$(join_by_comma "${sidenet_gpu_ids[@]}")" "$SIDENET_PI05_CMD" "SideNet+PI05 training")"
      log "Started SideNet+PI05 training pid=${sidenet_pid}"
      wait "$sidenet_pid"
      exit $?
    fi

    log "Free GPU count ${free_gpu_count} did not match launch policy. Sleeping ${CHECK_INTERVAL_SEC}s."
    sleep "$CHECK_INTERVAL_SEC"
  done
}

main "$@"
