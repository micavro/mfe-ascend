#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the model directory inside the container}"
QUESTIONS_FILE="${QUESTIONS_FILE:-$PROJECT_ROOT/data/experiments_design7/mixed_medium_first200.jsonl}"
DEVICE_IDS="${DEVICE_IDS:-0,1,2,3,4,5,6,7}"
EXPECTED_DEVICE_COUNT="${EXPECTED_DEVICE_COUNT:-}"
EXPECTED_REQUESTS="${EXPECTED_REQUESTS:-1400}"
POISSON_RATE="${POISSON_RATE:-0.13}"
ARRIVAL_SEED="${ARRIVAL_SEED:-20260709}"
ARRIVAL_BATCH_SIZE="${ARRIVAL_BATCH_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
OUTPUT_MAX_TOKENS="${OUTPUT_MAX_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
SCHEDULERS="${SCHEDULERS:-fcfs sjf rhsail}"
RESUME="${RESUME:-0}"

if [[ -z "$EXPECTED_DEVICE_COUNT" ]]; then
  IFS=',' read -r -a _device_array <<< "$DEVICE_IDS"
  EXPECTED_DEVICE_COUNT="${#_device_array[@]}"
fi

rate_tag="${POISSON_RATE//./}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/$(date +%Y%m%d-%H%M%S)-company-ascend-poisson${rate_tag}}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: model directory does not exist: $MODEL_PATH" >&2
  exit 2
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "ERROR: model config is missing: $MODEL_PATH/config.json" >&2
  exit 2
fi
if [[ ! -f "$QUESTIONS_FILE" ]]; then
  echo "ERROR: questions file does not exist: $QUESTIONS_FILE" >&2
  exit 2
fi

request_count="$(grep -cve '^[[:space:]]*$' "$QUESTIONS_FILE")"
if [[ "$request_count" != "$EXPECTED_REQUESTS" ]]; then
  echo "ERROR: questions file has $request_count non-empty rows, expected $EXPECTED_REQUESTS" >&2
  exit 2
fi

if [[ -d "$OUTPUT_ROOT" && -n "$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" && "$RESUME" != "1" ]]; then
  echo "ERROR: output root is not empty: $OUTPUT_ROOT" >&2
  echo "Use a new OUTPUT_ROOT, or set RESUME=1 to skip already complete scheduler runs." >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$OUTPUT_ROOT/runner.log") 2>&1
echo "$$" > "$OUTPUT_ROOT/runner.pid"
rm -f "$OUTPUT_ROOT/DONE" "$OUTPUT_ROOT/FAILED"

on_exit() {
  local status=$?
  if [[ "$status" -ne 0 ]]; then
    touch "$OUTPUT_ROOT/FAILED"
    echo "[$(date --iso-8601=seconds)] FAILED exit_code=$status"
  fi
}
trap on_exit EXIT

export MFE_ACCELERATOR=ascend
export MFE_DEVICE_IDS="$DEVICE_IDS"
export MFE_MODEL_PATH="$MODEL_PATH"
export MFE_DATA_DIR="${MFE_DATA_DIR:-$PROJECT_ROOT/data}"
export MFE_OFFLINE=1
export MFE_MAX_MODEL_LEN="$MAX_MODEL_LEN"
export MFE_OUTPUT_MAX_TOKENS="$OUTPUT_MAX_TOKENS"
export MFE_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION"
export MFE_ENABLE_PREFIX_CACHING="${MFE_ENABLE_PREFIX_CACHING:-0}"
export MFE_VLLM_LOG_LEVEL="${MFE_VLLM_LOG_LEVEL:-ERROR}"
export VLLM_TARGET_DEVICE=npu
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cat > "$OUTPUT_ROOT/run_config.txt" <<EOF
project_root=$PROJECT_ROOT
model_path=$MODEL_PATH
questions_file=$QUESTIONS_FILE
request_count=$request_count
device_ids=$DEVICE_IDS
expected_device_count=$EXPECTED_DEVICE_COUNT
schedulers=$SCHEDULERS
poisson_rate=$POISSON_RATE
arrival_seed=$ARRIVAL_SEED
arrival_batch_size=$ARRIVAL_BATCH_SIZE
max_model_len=$MAX_MODEL_LEN
output_max_tokens=$OUTPUT_MAX_TOKENS
gpu_memory_utilization=$GPU_MEMORY_UTILIZATION
prefix_caching=$MFE_ENABLE_PREFIX_CACHING
EOF

summary_is_complete() {
  local summary_path="$1"
  "$PYTHON_BIN" - "$summary_path" "$EXPECTED_REQUESTS" <<'PY'
import json
import sys

path, expected = sys.argv[1], int(sys.argv[2])
with open(path, "r", encoding="utf-8") as handle:
    summary = json.load(handle)
ok = (
    int(summary.get("count") or 0) == expected
    and int(summary.get("completed") or 0) == expected
    and float(summary.get("success_rate") or 0.0) >= 1.0 - 1e-9
)
raise SystemExit(0 if ok else 1)
PY
}

echo "[$(date --iso-8601=seconds)] preflight"
"$PYTHON_BIN" -m mfe.scripts.check_runtime_env \
  --accelerator ascend \
  --device-ids "$DEVICE_IDS" \
  --expected-device-count "$EXPECTED_DEVICE_COUNT" \
  --model-path "$MODEL_PATH" \
  --data-dir "$MFE_DATA_DIR" \
  --require-model-path \
  --require-data-dir \
  --require-homogeneous \
  --offline > "$OUTPUT_ROOT/preflight.json"

if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info > "$OUTPUT_ROOT/npu_smi_start.txt" 2>&1 || true
fi

read -r -a scheduler_array <<< "$SCHEDULERS"
for scheduler in "${scheduler_array[@]}"; do
  case "$scheduler" in
    fcfs|sjf|rhsail) ;;
    *)
      echo "ERROR: unsupported scheduler in this sweep: $scheduler" >&2
      exit 2
      ;;
  esac

  run_dir="$OUTPUT_ROOT/$scheduler"
  shopt -s nullglob
  existing_summaries=("$run_dir"/*_summary.json)
  shopt -u nullglob
  if [[ "$RESUME" == "1" && "${#existing_summaries[@]}" == "1" ]] && summary_is_complete "${existing_summaries[0]}"; then
    echo "[$(date --iso-8601=seconds)] SKIP complete scheduler=$scheduler"
    continue
  fi
  if [[ "${#existing_summaries[@]}" -gt "0" ]]; then
    echo "ERROR: existing scheduler output is not safely resumable: $run_dir" >&2
    exit 2
  fi

  mkdir -p "$run_dir"
  : > "$OUTPUT_ROOT/${scheduler}.log"
  echo "[$(date --iso-8601=seconds)] START scheduler=$scheduler"
  bash deploy/run_unified.sh company-ascend \
    --mode run \
    --model-path "$MODEL_PATH" \
    --questions-file "$QUESTIONS_FILE" \
    --scheduler "$scheduler" \
    --output-length medium \
    --output-max-tokens "$OUTPUT_MAX_TOKENS" \
    --repeat 1 \
    --arrival-mode poisson-burst \
    --arrival-batch-size "$ARRIVAL_BATCH_SIZE" \
    --poisson-rate "$POISSON_RATE" \
    --arrival-seed "$ARRIVAL_SEED" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --device-ids "$DEVICE_IDS" \
    --expected-device-count "$EXPECTED_DEVICE_COUNT" \
    --output-dir "$run_dir" \
    --offline \
    --skip-install 2>&1 | tee -a "$OUTPUT_ROOT/${scheduler}.log"

  shopt -s nullglob
  summaries=("$run_dir"/*_summary.json)
  shopt -u nullglob
  if [[ "${#summaries[@]}" != "1" ]] || ! summary_is_complete "${summaries[0]}"; then
    echo "ERROR: scheduler=$scheduler did not finish $EXPECTED_REQUESTS/$EXPECTED_REQUESTS at 100%" >&2
    exit 1
  fi
  echo "[$(date --iso-8601=seconds)] DONE scheduler=$scheduler"
done

error_pattern='out of memory|(^|[^[:alnum:]_])OOM([^[:alnum:]_]|$)|Traceback|CUDA error|context length.*(exceed|error)|maximum context.*(exceed|error)|KV cache.*(failed|error|insufficient)|ACL.*[Ee]rror|HCCL.*[Ee]rror'
scheduler_logs=()
for scheduler in "${scheduler_array[@]}"; do
  scheduler_logs+=("$OUTPUT_ROOT/${scheduler}.log")
done
if grep -Eain "$error_pattern" "${scheduler_logs[@]}" > "$OUTPUT_ROOT/error_scan.txt" 2>/dev/null; then
  echo "ERROR: fatal error keywords found; see $OUTPUT_ROOT/error_scan.txt" >&2
  cat "$OUTPUT_ROOT/error_scan.txt" >&2
  exit 1
fi
: > "$OUTPUT_ROOT/error_scan.txt"

"$PYTHON_BIN" -m mfe.scripts.summarize_scheduler_runs "$OUTPUT_ROOT" \
  --schedulers "${scheduler_array[@]}" \
  --expected-count "$EXPECTED_REQUESTS"

if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info > "$OUTPUT_ROOT/npu_smi_end.txt" 2>&1 || true
fi

touch "$OUTPUT_ROOT/DONE"
echo "[$(date --iso-8601=seconds)] ALL DONE"
