# 进入现有 vLLM Docker 后运行实验

两台服务器都已经有配置好 Python、vLLM Ascend，并映射了 8 张 NPU 的 Docker。这里不创建新容器；进入现有容器后，可以一次注册 FCFS、SJF、RH-SAIL 三个实验并让它们顺序执行，也可以逐个手动运行以便排障。

- 服务器 A：`0.12 req/s`，顺序运行 FCFS、SJF、RH-SAIL。
- 服务器 B：`0.15 req/s`，顺序运行 FCFS、SJF、RH-SAIL。
- Docker 虽然映射 8 张卡，实验只使用 `0,1,2,3,4` 五张卡。
- 假设容器内项目是 `/x/mfe-ascend`，模型是 `/x/Meta-Llmama-3.1-8B-Instruct`。

如果实际挂载路径或模型目录名不同，只修改开头的 `X` 和 `MODEL_PATH`。

## 1. 下载 ZIP 并通过 SSH 上传

在自己的 Windows PowerShell 下载默认 `main`：

```powershell
$zip = "$HOME\Downloads\mfe-ascend.zip"
Invoke-WebRequest `
  -Uri "https://github.com/micavro/mfe-ascend/archive/refs/heads/main.zip" `
  -OutFile $zip
python -m zipfile -t $zip
```

最后一行必须输出 `Done testing`。然后上传到两台服务器中映射为容器 `/x` 的宿主机目录。假设宿主机也是 `/x`：

```powershell
scp -P 22 $zip USER@SERVER_A:/x/
scp -P 22 $zip USER@SERVER_B:/x/
```

## 2. 解压到挂载目录并进入 Docker

在两台服务器宿主机执行：

```bash
cd /x

if [ -d mfe-ascend ]; then
  mv mfe-ascend "mfe-ascend.backup.$(date +%Y%m%d-%H%M%S)"
fi

python -m zipfile -e mfe-ascend.zip .
mv mfe-ascend-main mfe-ascend

docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
docker exec -it <现有容器名> bash
```

以下所有命令都在 Docker 内执行。

## 3. 设置路径、速率和五卡环境

服务器 A 使用 `RATE=0.12`，服务器 B 改成 `RATE=0.15`：

```bash
export X=/x
export PROJECT=$X/mfe-ascend
export MODEL_PATH=$X/Meta-Llmama-3.1-8B-Instruct
export QUESTIONS_FILE=$PROJECT/data/experiments_design7/mixed_medium_first200.jsonl

export RATE=0.12
export DEVICE_IDS=0,1,2,3,4
export EXPECTED_DEVICE_COUNT=5

export MFE_ACCELERATOR=ascend
export MFE_DEVICE_IDS="$DEVICE_IDS"
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE_IDS"
export NPU_VISIBLE_DEVICES="$DEVICE_IDS"
export VLLM_TARGET_DEVICE=npu
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export MFE_ENABLE_PREFIX_CACHING=0
export MFE_VLLM_ENFORCE_EAGER=1
export MFE_VLLM_LOG_LEVEL=ERROR
export MFE_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "$PROJECT"
```

检查文件、数据、活动服务和 NPU：

```bash
test -f "$MODEL_PATH/config.json" && echo MODEL_OK
test -f "$QUESTIONS_FILE" && echo DATA_OK
grep -cve '^[[:space:]]*$' "$QUESTIONS_FILE"
npu-smi info

ps -eo pid,args |
  grep -E '[v]llm[[:space:]]+serve|vllm\.entrypoints\.openai\.api_server' |
  grep -v grep || true

python -c "import torch,torch_npu,vllm,vllm_ascend; print('visible_npu=',torch.npu.device_count()); print('vllm=',vllm.__version__)"
```

必须满足：请求数为 `1400`、`visible_npu=5`、模型和数据均为 `OK`。如果进程检查显示活跃的 `vllm serve`，不能在同一批 NPU 上继续实验；不要自动停止不属于自己的进程。

## 4. 预检并注册三个策略

先运行环境预检：

```bash
bash deploy/run_unified.sh company-ascend \
  --mode check \
  --model-path "$MODEL_PATH" \
  --questions-file "$QUESTIONS_FILE" \
  --device-ids "$DEVICE_IDS" \
  --expected-device-count 5 \
  --offline \
  --skip-install
```

退出码必须为 0。接下来二选一：推荐使用 4A 一次注册三个策略；需要逐步排障时使用 4B。

### 4A. 一次注册三个策略，顺序执行

这会一次登记 FCFS、SJF、RH-SAIL，但不会同时运行。脚本等 FCFS 完成后自动启动 SJF，最后启动 RH-SAIL，三者共用同一批五张 NPU。

```bash
RATE_TAG="${RATE/./}"
export RUN_ROOT="$X/outputs/$(date +%Y%m%d-%H%M%S)-rate${RATE_TAG}-fcfs-sjf-rhsail"
export LAUNCH_LOG="$X/outputs/launcher-$(basename "$RUN_ROOT").log"

export EXPECTED_REQUESTS=1400
export POISSON_RATE="$RATE"
export ARRIVAL_SEED=20260709
export ARRIVAL_BATCH_SIZE=1
export MAX_MODEL_LEN=32768
export OUTPUT_MAX_TOKENS=2048
export GPU_MEMORY_UTILIZATION=0.75
export SCHEDULERS="fcfs sjf rhsail"
export OUTPUT_ROOT="$RUN_ROOT"

mkdir -p "$X/outputs"
nohup bash deploy/run_company_ascend_sweep.sh > "$LAUNCH_LOG" 2>&1 &
export BATCH_PID=$!
echo "$BATCH_PID" > "${LAUNCH_LOG%.log}.pid"

echo "BATCH_PID=$BATCH_PID"
echo "RUN_ROOT=$RUN_ROOT"
echo "LAUNCH_LOG=$LAUNCH_LOG"
```

不要提前执行 `mkdir -p "$RUN_ROOT"`，也不要把 launcher 日志写进 `RUN_ROOT`；正式脚本会自己创建空结果目录。

确认批次已经开始：

```bash
cat "$LAUNCH_LOG"
sleep 2
cat "$RUN_ROOT/run_config.txt"
tail -F "$RUN_ROOT/runner.log"
```

退出 `tail` 按 `Ctrl+C`，不会停止后台实验。批量模式会自动检查每个策略的 `1400/1400`、扫描错误、生成最终简报并创建 `DONE`。

### 4B. 逐个手动运行，作为排障备用

手动模式需要创建结果目录和共同参数：

```bash
RATE_TAG="${RATE/./}"
export RUN_ROOT="$X/outputs/$(date +%Y%m%d-%H%M%S)-rate${RATE_TAG}-fcfs-sjf-rhsail"
mkdir -p "$RUN_ROOT"
set -o pipefail

COMMON_ARGS=(
  company-ascend
  --mode run
  --model-path "$MODEL_PATH"
  --questions-file "$QUESTIONS_FILE"
  --output-length medium
  --output-max-tokens 2048
  --repeat 1
  --arrival-mode poisson-burst
  --arrival-batch-size 1
  --poisson-rate "$RATE"
  --arrival-seed 20260709
  --max-model-len 32768
  --gpu-memory-utilization 0.75
  --device-ids "$DEVICE_IDS"
  --expected-device-count 5
  --offline
  --skip-install
)

echo "RATE=$RATE"
echo "RUN_ROOT=$RUN_ROOT"
```

先运行 FCFS，必须等它结束：

```bash
bash deploy/run_unified.sh "${COMMON_ARGS[@]}" \
  --scheduler fcfs --output-dir "$RUN_ROOT/fcfs" \
  2>&1 | tee "$RUN_ROOT/fcfs.log"
test "${PIPESTATUS[0]}" -eq 0 || { echo FCFS_FAILED; exit 1; }
cat "$RUN_ROOT/fcfs/brief_summary.txt"
```

再运行 SJF：

```bash
bash deploy/run_unified.sh "${COMMON_ARGS[@]}" \
  --scheduler sjf --output-dir "$RUN_ROOT/sjf" \
  2>&1 | tee "$RUN_ROOT/sjf.log"
test "${PIPESTATUS[0]}" -eq 0 || { echo SJF_FAILED; exit 1; }
cat "$RUN_ROOT/sjf/brief_summary.txt"
```

最后运行 RH-SAIL：

```bash
bash deploy/run_unified.sh "${COMMON_ARGS[@]}" \
  --scheduler rhsail --output-dir "$RUN_ROOT/rhsail" \
  2>&1 | tee "$RUN_ROOT/rhsail.log"
test "${PIPESTATUS[0]}" -eq 0 || { echo RHSAIL_FAILED; exit 1; }
cat "$RUN_ROOT/rhsail/brief_summary.txt"
```

三个命令只能顺序执行，不能同时占用同一批 NPU。不要同时使用 4A 和 4B。手动模式长时间运行时建议在 `tmux` 或 `screen` 会话中操作。

## 5. 检查完成并生成简报

如果使用 4A 批量注册，脚本已经自动完成错误扫描和简报生成，只需验收：

```bash
grep -E 'START scheduler=|DONE scheduler=|ALL DONE|FAILED' "$RUN_ROOT/runner.log"
test -f "$RUN_ROOT/DONE" && echo ALL_COMPLETE
test ! -f "$RUN_ROOT/FAILED" && echo NO_FAILED_MARKER
test ! -s "$RUN_ROOT/error_scan.txt" && echo NO_FATAL_KEYWORDS
cat "$RUN_ROOT/final_brief.txt"
```

如果使用 4B 逐个手动运行，三个策略结束后先扫描错误：

```bash
grep -Eain \
  'out of memory|OOM|Traceback|CUDA error|context length|maximum context|KV cache|ACL.*[Ee]rror|HCCL.*[Ee]rror' \
  "$RUN_ROOT"/*.log > "$RUN_ROOT/error_scan.txt" || true

test ! -s "$RUN_ROOT/error_scan.txt" || {
  cat "$RUN_ROOT/error_scan.txt"
  echo DO_NOT_MARK_DONE
  exit 1
}
```

生成严格的三策略简报：

```bash
python -m mfe.scripts.summarize_scheduler_runs "$RUN_ROOT" \
  --schedulers fcfs sjf rhsail \
  --expected-count 1400

touch "$RUN_ROOT/DONE"
cat "$RUN_ROOT/final_brief.txt"
```

FCFS、SJF、RH-SAIL 都必须显示 `done=1400/1400 ok=100.0%`，并且存在 `DONE`、不存在错误关键词。

## 6. 结果位置

因为输出写在 Docker 挂载目录 `/x`，退出 Docker 后宿主机可以直接访问，不需要 `docker cp`：

```text
/x/outputs/<时间>-rate012-fcfs-sjf-rhsail/
/x/outputs/<时间>-rate015-fcfs-sjf-rhsail/
```

主要文件：

```text
fcfs/、sjf/、rhsail/       每个策略的 detail JSON、summary JSON、brief_summary
fcfs.log、sjf.log、rhsail.log
error_scan.txt             必须为空
final_brief.txt            最简结果
final_brief.md             Markdown 表格
final_brief.csv            汇总数据
DONE                       全部成功标记
```

无法带出公司数据时，至少拍下：

```bash
cat "$RUN_ROOT/final_brief.txt"
cat "$RUN_ROOT/fcfs/brief_summary.txt"
cat "$RUN_ROOT/sjf/brief_summary.txt"
cat "$RUN_ROOT/rhsail/brief_summary.txt"
```
