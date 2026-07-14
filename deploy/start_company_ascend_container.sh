#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:?Set IMAGE to a locally available, company-approved vllm-ascend image}"
HOST_WORK="${HOST_WORK:?Set HOST_WORK to the host directory containing mfe-ascend and models}"
NPU_DEVICE_IDS="${NPU_DEVICE_IDS:-0,1,2,3,4,5,6,7}"
CONTAINER_NAME="${CONTAINER_NAME:-mfe-ascend-company}"
CONTAINER_WORK="${CONTAINER_WORK:-/data/mfe}"
SHM_SIZE="${SHM_SIZE:-16g}"

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found" >&2; exit 2; }
command -v npu-smi >/dev/null 2>&1 || { echo "ERROR: npu-smi not found on the host" >&2; exit 2; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "ERROR: Docker image is not available locally: $IMAGE" >&2
  exit 2
}
[[ -d "$HOST_WORK/mfe-ascend" ]] || {
  echo "ERROR: project directory is missing: $HOST_WORK/mfe-ascend" >&2
  exit 2
}

required_devices=(/dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc)
IFS=',' read -r -a device_ids <<< "$NPU_DEVICE_IDS"
for id in "${device_ids[@]}"; do
  required_devices+=("/dev/davinci${id}")
done
for path in "${required_devices[@]}"; do
  [[ -e "$path" ]] || { echo "ERROR: required device is missing: $path" >&2; exit 2; }
done

required_mounts=(
  /usr/local/dcmi
  /usr/local/bin/npu-smi
  /usr/local/Ascend/driver/lib64
  /usr/local/Ascend/driver/version.info
  /etc/ascend_install.info
)
for path in "${required_mounts[@]}"; do
  [[ -e "$path" ]] || { echo "ERROR: required host path is missing: $path" >&2; exit 2; }
done

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "ERROR: container already exists: $CONTAINER_NAME" >&2
  echo "Use it with: docker exec -it $CONTAINER_NAME bash" >&2
  exit 2
fi

args=(
  docker run -d
  --name "$CONTAINER_NAME"
  --network host
  --shm-size "$SHM_SIZE"
)
for path in "${required_devices[@]}"; do
  args+=(--device "$path")
done
args+=(
  -v /usr/local/dcmi:/usr/local/dcmi
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info
  -v /etc/ascend_install.info:/etc/ascend_install.info
  -v "$HOST_WORK:$CONTAINER_WORK"
  -w "$CONTAINER_WORK/mfe-ascend"
)
if [[ -e /usr/local/Ascend/driver/tools/hccn_tool ]]; then
  args+=(-v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool)
fi
args+=("$IMAGE" sleep infinity)

"${args[@]}"
echo "container started: $CONTAINER_NAME"
echo "enter it with: docker exec -it $CONTAINER_NAME bash"
