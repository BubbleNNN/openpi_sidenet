#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-${REPO_ROOT}/launcher_logs}"
WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p "$LAUNCH_LOG_DIR"

# Configuration
# The script will prepend CUDA_VISIBLE_DEVICES automatically.
SIDENET_PI05_CMD="cd \"$REPO_ROOT\" && torchrun --standalone --nnodes=1 --nproc_per_node=8 -m sidenet.train pi05_with_sidenet --exp-name \"water_hose_sidenet_pi05_${RUN_TAG}\""

# Matching policy:
# - exact: launch only when free GPU count is exactly 8
# - at_least: launch when free GPU count is >= 8
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
  local log_slug="$4"
  local log_file="${LAUNCH_LOG_DIR}/${log_slug}_${RUN_TAG}.log"
  local launch_ld_library_path="${LD_LIBRARY_PATH:-}"

  echo "[$(timestamp)] Launching ${tag} on GPUs: ${gpu_ids_csv}" >&2
  echo "[$(timestamp)] ${tag} log file: ${log_file}" >&2
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    launch_ld_library_path="${CONDA_PREFIX}/lib${launch_ld_library_path:+:${launch_ld_library_path}}"
  fi
  CUDA_VISIBLE_DEVICES="$gpu_ids_csv" \
  LD_LIBRARY_PATH="$launch_ld_library_path" \
  PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}" \
  WANDB_MODE="$WANDB_MODE" \
  bash -lc "$command_string" >"$log_file" 2>&1 &
  echo "$!"
}

main() {
  require_command nvidia-smi
  require_command torchrun
  validate_configuration

  log "Waiting for free GPUs. MATCH_MODE=${MATCH_MODE}, MEM<=${GPU_MEM_USED_THRESHOLD_MB}MB, UTIL<=${GPU_UTIL_THRESHOLD_PERCENT}%"
  log "WandB mode: ${WANDB_MODE}"

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
      sidenet_gpu_ids=("${free_gpu_ids[@]:0:8}")

      if (( ${#sidenet_gpu_ids[@]} < 8 )); then
        log "Matched 8-GPU policy, but failed to allocate 8 GPUs. Retrying."
        sleep "$CHECK_INTERVAL_SEC"
        continue
      fi

      sidenet_pid="$(launch_command "$(join_by_comma "${sidenet_gpu_ids[@]}")" "$SIDENET_PI05_CMD" "SideNet+PI05 training" "sidenet_pi05")"
      log "Started SideNet+PI05 training pid=${sidenet_pid}"
      wait "$sidenet_pid"
      exit $?
    fi

    log "Free GPU count ${free_gpu_count} did not match launch policy. Sleeping ${CHECK_INTERVAL_SEC}s."
    sleep "$CHECK_INTERVAL_SEC"
  done
}

main "$@"
