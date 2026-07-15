#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST_PROJECT="${HOST_PROJECT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
CONTAINER_NAME="${CONTAINER_NAME:?Set CONTAINER_NAME to the running vLLM container name}"
CONTAINER_PROJECT="${CONTAINER_PROJECT:-/workspace/mfe-ascend}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the model directory inside the container}"
POISSON_RATE="${POISSON_RATE:?Set POISSON_RATE to 0.12 or 0.15}"
DEVICE_IDS="${DEVICE_IDS:-0,1,2,3,4}"
EXPECTED_DEVICE_COUNT="${EXPECTED_DEVICE_COUNT:-5}"
EXPECTED_REQUESTS="${EXPECTED_REQUESTS:-1400}"
QUESTIONS_FILE="${QUESTIONS_FILE:-$CONTAINER_PROJECT/data/experiments_design7/mixed_medium_first200.jsonl}"
SCHEDULERS="${SCHEDULERS:-fcfs sjf rhsail}"
ARRIVAL_SEED="${ARRIVAL_SEED:-20260709}"
ARRIVAL_BATCH_SIZE="${ARRIVAL_BATCH_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
OUTPUT_MAX_TOKENS="${OUTPUT_MAX_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
COPY_PROJECT="${COPY_PROJECT:-1}"
SKIP_ACTIVE_SERVICE_CHECK="${SKIP_ACTIVE_SERVICE_CHECK:-0}"

case "$POISSON_RATE" in
  0.12|0.15) ;;
  *)
    echo "ERROR: this two-machine experiment expects POISSON_RATE=0.12 or 0.15" >&2
    exit 2
    ;;
esac

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found on the host" >&2; exit 2; }
[[ -d "$HOST_PROJECT" ]] || { echo "ERROR: host project does not exist: $HOST_PROJECT" >&2; exit 2; }

running="$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)"
if [[ "$running" != "true" ]]; then
  echo "ERROR: container is not running: $CONTAINER_NAME" >&2
  exit 2
fi

if [[ "$SKIP_ACTIVE_SERVICE_CHECK" != "1" ]]; then
  active_service="$(
    docker exec "$CONTAINER_NAME" sh -lc \
      "ps -eo pid,args | grep -E '[v]llm[[:space:]]+serve|vllm\.entrypoints\.openai\.api_server' || true"
  )"
  if [[ -n "$active_service" ]]; then
    echo "ERROR: an active vLLM API service is already running in the container:" >&2
    echo "$active_service" >&2
    echo "Do not run MFE on the same five NPUs until that service is stopped or moved." >&2
    exit 2
  fi
fi

IFS=',' read -r -a device_array <<< "$DEVICE_IDS"
if [[ "${#device_array[@]}" != "$EXPECTED_DEVICE_COUNT" ]]; then
  echo "ERROR: DEVICE_IDS contains ${#device_array[@]} devices, expected $EXPECTED_DEVICE_COUNT" >&2
  exit 2
fi
for id in "${device_array[@]}"; do
  docker exec "$CONTAINER_NAME" test -e "/dev/davinci${id}" || {
    echo "ERROR: /dev/davinci${id} is not available in the container" >&2
    exit 2
  }
done

if [[ "$COPY_PROJECT" == "1" ]]; then
  docker exec "$CONTAINER_NAME" mkdir -p "$CONTAINER_PROJECT"
  docker cp "$HOST_PROJECT/." "$CONTAINER_NAME:$CONTAINER_PROJECT/"
fi

docker exec "$CONTAINER_NAME" test -f "$CONTAINER_PROJECT/deploy/run_company_ascend_sweep.sh" || {
  echo "ERROR: MFE project is missing inside the container: $CONTAINER_PROJECT" >&2
  exit 2
}
docker exec "$CONTAINER_NAME" test -f "$MODEL_PATH/config.json" || {
  echo "ERROR: model config is missing inside the container: $MODEL_PATH/config.json" >&2
  exit 2
}
docker exec "$CONTAINER_NAME" test -f "$QUESTIONS_FILE" || {
  echo "ERROR: questions file is missing inside the container: $QUESTIONS_FILE" >&2
  exit 2
}

request_count="$(
  docker exec "$CONTAINER_NAME" python -c \
    'import sys; print(sum(1 for line in open(sys.argv[1], encoding="utf-8") if line.strip()))' \
    "$QUESTIONS_FILE"
)"
if [[ "$request_count" != "$EXPECTED_REQUESTS" ]]; then
  echo "ERROR: questions file has $request_count rows, expected $EXPECTED_REQUESTS" >&2
  exit 2
fi

visible_count="$(
  docker exec \
    -e MFE_DEVICE_IDS="$DEVICE_IDS" \
    -e ASCEND_RT_VISIBLE_DEVICES="$DEVICE_IDS" \
    -e NPU_VISIBLE_DEVICES="$DEVICE_IDS" \
    "$CONTAINER_NAME" \
    python -c 'import torch, torch_npu; print(torch.npu.device_count())' | tail -n 1
)"
if [[ "$visible_count" != "$EXPECTED_DEVICE_COUNT" ]]; then
  echo "ERROR: torch sees $visible_count NPUs after visibility filtering, expected $EXPECTED_DEVICE_COUNT" >&2
  exit 2
fi

rate_tag="${POISSON_RATE//./}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/mfe-results/$(date +%Y%m%d-%H%M%S)-rate${rate_tag}-fcfs-sjf-rhsail}"
docker exec "$CONTAINER_NAME" mkdir -p "$(dirname "$OUTPUT_ROOT")"

docker exec -d \
  -e MODEL_PATH="$MODEL_PATH" \
  -e QUESTIONS_FILE="$QUESTIONS_FILE" \
  -e DEVICE_IDS="$DEVICE_IDS" \
  -e EXPECTED_DEVICE_COUNT="$EXPECTED_DEVICE_COUNT" \
  -e EXPECTED_REQUESTS="$EXPECTED_REQUESTS" \
  -e POISSON_RATE="$POISSON_RATE" \
  -e ARRIVAL_SEED="$ARRIVAL_SEED" \
  -e ARRIVAL_BATCH_SIZE="$ARRIVAL_BATCH_SIZE" \
  -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  -e OUTPUT_MAX_TOKENS="$OUTPUT_MAX_TOKENS" \
  -e GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  -e SCHEDULERS="$SCHEDULERS" \
  -e MFE_ENABLE_PREFIX_CACHING=0 \
  -e OUTPUT_ROOT="$OUTPUT_ROOT" \
  "$CONTAINER_NAME" \
  bash -lc "cd '$CONTAINER_PROJECT' && bash deploy/run_company_ascend_sweep.sh"

echo "STARTED container=$CONTAINER_NAME rate=$POISSON_RATE devices=$DEVICE_IDS"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "monitor: docker exec $CONTAINER_NAME tail -n 100 -f $OUTPUT_ROOT/runner.log"
echo "result:  docker exec $CONTAINER_NAME cat $OUTPUT_ROOT/final_brief.txt"
echo "copy:    docker cp $CONTAINER_NAME:$OUTPUT_ROOT /data/mfe/results/"
