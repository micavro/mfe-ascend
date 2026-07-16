# 进入现有 vLLM Docker 后运行实验

两台服务器都已经有配置好 Python、vLLM Ascend，并映射了 8 张 NPU 的 Docker。这里不创建新容器；进入现有容器后，可以一次注册 FCFS、SJF、RH-SAIL 三个实验并让它们顺序执行，也可以逐个手动运行以便排障。

- 服务器 A：`0.12 req/s`，顺序运行 FCFS、SJF、RH-SAIL。
- 服务器 B：`0.03 req/s`，顺序运行 FCFS、SJF、RH-SAIL。
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

服务器 A 使用 `RATE=0.12`，服务器 B 改成 `RATE=0.03`。两台机器都只更换为 `first50` 数据，其他实验参数保持一致：

```bash
export X=/x
export PROJECT=$X/mfe-ascend
export MODEL_PATH=$X/Meta-Llmama-3.1-8B-Instruct
export QUESTIONS_FILE=$PROJECT/data/experiments_design7/mixed_medium_first50.jsonl

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

必须满足：请求数为 `350`、`visible_npu=5`、模型和数据均为 `OK`。如果进程检查显示活跃的 `vllm serve`，不能在同一批 NPU 上继续实验；不要自动停止不属于自己的进程。

`0.03 req/s` 的负载较轻，但运行时间并不短：350 个请求的期望到达时长约为 `350 / 0.03 = 11667s`，即每个策略约 3.24 小时，三个策略顺序运行仅到达过程就约 9.7 小时。`0.12 req/s` 每个策略的期望到达时长约 48.6 分钟。

### 3A. 注销已经注册的旧 first200/1400 批次

修改 `QUESTIONS_FILE` 不会改变已经启动的进程。如果旧的 1400 请求批次仍在运行，不能直接在同一组 NPU 上启动新的 350 请求批次。先列出符合 `first200 + 1400` 的旧结果目录：

```bash
for config in /x/outputs/*/run_config.txt; do
  if grep -qx 'request_count=1400' "$config" && \
     grep -q 'questions_file=.*mixed_medium_first200.jsonl' "$config"; then
    dirname "$config"
  fi
done
```

从输出中选择仍在运行的旧批次；不要选择新的 `first50` 目录：

```bash
export OLD_RUN_ROOT=/x/outputs/<旧的 first200/1400 结果目录>
cat "$OLD_RUN_ROOT/run_config.txt"
cat "$OLD_RUN_ROOT/runner.pid"
```

必须人工确认配置中包含旧数据文件 `mixed_medium_first200.jsonl` 和 `request_count=1400`。然后运行专用注销脚本：

```bash
cd /x/mfe-ascend
bash deploy/cancel_company_ascend_sweep.sh "$OLD_RUN_ROOT"
```

脚本会再次校验目录、数据文件、请求数、runner PID 和进程命令，并显示该 runner 的完整子进程树。只有手动输入 `CANCEL-1400` 才会发出终止信号。它先等待进程正常退出，必要时只强制结束此前确认的同一进程树；不会执行 `pkill python`、`pkill vllm`，也不会影响其他容器或其他任务。旧目录不会删除，而是创建 `CANCELLED` 和 `FAILED` 标记。

注销后检查旧进程已经消失、五张目标 NPU 已经释放：

```bash
OLD_RUNNER_PID="$(cat "$OLD_RUN_ROOT/runner.pid")"
kill -0 "$OLD_RUNNER_PID" 2>/dev/null && echo STILL_RUNNING || echo OLD_RUN_STOPPED
test -f "$OLD_RUN_ROOT/CANCELLED" && echo OLD_RUN_CANCELLED
npu-smi info
```

必须看到 `OLD_RUN_STOPPED`，并确认设备 `0,1,2,3,4` 上没有旧实验进程，才能继续注册新的 first50 批次。不要直接删除旧输出目录，也不要使用无范围的 `killall` 或 `pkill`。

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
export RUN_ROOT="$X/outputs/$(date +%Y%m%d-%H%M%S)-first50-rate${RATE_TAG}-fcfs-sjf-rhsail"
export LAUNCH_LOG="$X/outputs/launcher-$(basename "$RUN_ROOT").log"

export EXPECTED_REQUESTS=350
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

退出 `tail` 按 `Ctrl+C`，不会停止后台实验。批量模式会自动检查每个策略的 `350/350`、扫描错误、生成最终简报并创建 `DONE`。

### 4B. 逐个手动运行，作为排障备用

手动模式需要创建结果目录和共同参数：

```bash
RATE_TAG="${RATE/./}"
export RUN_ROOT="$X/outputs/$(date +%Y%m%d-%H%M%S)-first50-rate${RATE_TAG}-fcfs-sjf-rhsail"
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

#### 4B.1. FCFS 已手动启动后，注册剩余两个策略

不能在 FCFS 仍占用 NPU 时直接启动 SJF 或 RH-SAIL。正确做法是等 FCFS 结束后，以 `RESUME=1` 启动完整批次：批处理脚本会验证并跳过已经成功完成的 FCFS，然后顺序运行 SJF 和 RH-SAIL，最后生成包含三个策略的统一简报。

后续任务的 `POISSON_RATE`、prefix caching、eager/graph 模式、模型长度和显存比例必须与当前 FCFS 完全相同，否则三者不可直接比较。

如果 FCFS 已经结束并且 `brief_summary.txt` 显示 `350/350、100%`，在原终端执行：

```bash
cd "$PROJECT"

export POISSON_RATE="$RATE"
export OUTPUT_ROOT="$RUN_ROOT"
export SCHEDULERS="fcfs sjf rhsail"
export RESUME=1
export REMAIN_LOG="$X/outputs/remaining-$(basename "$RUN_ROOT").log"

nohup bash deploy/run_company_ascend_sweep.sh \
  > "$REMAIN_LOG" 2>&1 < /dev/null &
export REMAIN_PID=$!

echo "REMAIN_PID=$REMAIN_PID"
echo "REMAIN_LOG=$REMAIN_LOG"
tail -F "$REMAIN_LOG"
```

日志中应先出现 `SKIP complete scheduler=fcfs`，随后出现 `START scheduler=sjf`。如果 FCFS 不完整，批处理脚本会拒绝续跑并创建 `FAILED`，不会把残缺结果当成成功结果。

如果 FCFS 仍在前台运行，打开第二个终端进入同一个 Docker，重新执行第 3 节以恢复路径、速率和 NPU 环境变量，然后指向当前结果目录：

```bash
cd /x/mfe-ascend
export RUN_ROOT=/x/outputs/<当前 FCFS 所在的结果目录>

FCFS_PID="$(pgrep -fo "run_unified.sh.*--scheduler fcfs.*$RUN_ROOT/fcfs")"
test -n "$FCFS_PID" || {
  echo FCFS_PROCESS_NOT_FOUND
  exit 1
}
ps -fp "$FCFS_PID"
```

先核对 FCFS 实际继承的关键环境变量；后续两个策略必须使用相同值：

```bash
FCFS_PY_PID="$(pgrep -P "$FCFS_PID" -f python | head -1)"
test -n "$FCFS_PY_PID" && \
  tr '\0' '\n' < "/proc/$FCFS_PY_PID/environ" \
  | grep -E '^(MFE_ENABLE_PREFIX_CACHING|MFE_VLLM_ENFORCE_EAGER|MFE_GPU_MEMORY_UTILIZATION|MFE_MAX_MODEL_LEN)='
```

注册等待任务。下面的 prefix caching 和 eager 值以第 3 节启动 FCFS 的默认配置为例；如果 FCFS 使用了其他值，必须同步修改：

```bash
export POISSON_RATE="$RATE"
export QUEUE_SCRIPT="$RUN_ROOT/queue_remaining.sh"
export REMAIN_LOG="$X/outputs/remaining-$(basename "$RUN_ROOT").log"

cat > "$QUEUE_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail

while kill -0 "$FCFS_PID" 2>/dev/null; do
  sleep 30
done
sleep 20

cd "$PROJECT"
export MODEL_PATH="$MODEL_PATH"
export QUESTIONS_FILE="$QUESTIONS_FILE"
export DEVICE_IDS="$DEVICE_IDS"
export EXPECTED_DEVICE_COUNT=5
export EXPECTED_REQUESTS=350
export POISSON_RATE="$POISSON_RATE"
export ARRIVAL_SEED=20260709
export ARRIVAL_BATCH_SIZE=1
export MAX_MODEL_LEN=32768
export OUTPUT_MAX_TOKENS=2048
export GPU_MEMORY_UTILIZATION=0.75

# 必须与已经运行的 FCFS 保持一致。
export MFE_ENABLE_PREFIX_CACHING=0
export MFE_VLLM_ENFORCE_EAGER=1

export OUTPUT_ROOT="$RUN_ROOT"
export SCHEDULERS="fcfs sjf rhsail"
export RESUME=1

exec bash deploy/run_company_ascend_sweep.sh
EOF

chmod +x "$QUEUE_SCRIPT"
nohup "$QUEUE_SCRIPT" > "$REMAIN_LOG" 2>&1 < /dev/null &
export QUEUE_PID=$!

echo "QUEUE_PID=$QUEUE_PID"
echo "REMAIN_LOG=$REMAIN_LOG"
tail -F "$REMAIN_LOG"
```

这个等待任务只轮询当前 FCFS 的 PID，不会停止或修改它。FCFS 进程结束后会额外等待 20 秒释放 NPU，再由批处理脚本严格验证 FCFS summary；验证成功才会继续 SJF 和 RH-SAIL。

### 4C. 运行中检查是否健康

如果重新进入了 Docker，先找回最新一次实验目录；当前终端仍有 `RUN_ROOT` 时不会覆盖它：

```bash
export X="${X:-/x}"
if [[ -z "${RUN_ROOT:-}" ]]; then
  export RUN_ROOT="$(ls -dt "$X"/outputs/*-fcfs-sjf-rhsail 2>/dev/null | head -1)"
fi
test -n "$RUN_ROOT" && test -d "$RUN_ROOT" || {
  echo "RUN_ROOT_NOT_FOUND"
  exit 1
}
echo "RUN_ROOT=$RUN_ROOT"
```

先确认实际生效的参数。服务器 A 的 `poisson_rate` 应为 `0.12`，服务器 B 应为 `0.03`：

```bash
cat "$RUN_ROOT/run_config.txt"
```

重点检查 `request_count=350`、`device_ids=0,1,2,3,4`、`expected_device_count=5`、`schedulers=fcfs sjf rhsail`、`arrival_batch_size=1`、`max_model_len=32768`、`output_max_tokens=2048`、`gpu_memory_utilization=0.75` 和 `prefix_caching=0`。

查看当前策略、已经完成的策略和各策略的最新进度：

```bash
grep -E 'START scheduler=|DONE scheduler=|ALL DONE|FAILED' \
  "$RUN_ROOT/runner.log"

for scheduler in fcfs sjf rhsail; do
  if test -f "$RUN_ROOT/$scheduler.log"; then
    printf '%-8s ' "$scheduler"
    grep -aoE 'waiting: [0-9]+/350 completed' \
      "$RUN_ROOT/$scheduler.log" | tail -1
  fi
done
```

正常情况下，只能有一个已经 `START` 但尚未 `DONE` 的策略。进度应整体持续增长；单个长请求可能让数字短暂停留几分钟，需要结合日志更新时间和 NPU 活动一起判断。

检查批次进程和日志是否还在更新：

```bash
if test -f "$RUN_ROOT/DONE"; then
  echo ALL_COMPLETE
else
  RUNNER_PID="$(cat "$RUN_ROOT/runner.pid")"
  ps -fp "$RUNNER_PID"
  kill -0 "$RUNNER_PID" && echo RUNNER_ALIVE
fi
stat -c 'runner.log updated: %y' "$RUN_ROOT/runner.log"
tail -n 30 "$RUN_ROOT/runner.log"
```

实验进行中应显示 `RUNNER_ALIVE`。全部完成后 runner 正常退出，此时以 `DONE` 为准。持续查看日志可运行 `tail -F "$RUN_ROOT/runner.log"`；按 `Ctrl+C` 只退出查看，不会停止后台实验。

在另一个终端观察五张 NPU：

```bash
watch -n 5 npu-smi info
```

设备 0--4 应有实验进程和稳定的显存占用，利用率可以随泊松到达上下波动。短暂空闲正常；显存持续逼近上限、进程消失或出现 HCCL/ACL 告警则不正常。

随时扫描严重错误：

```bash
grep -Eain \
  'out of memory|(^|[^[:alnum:]_])OOM([^[:alnum:]_]|$)|Traceback|CUDA error|context length.*(exceed|error)|maximum context.*(exceed|error)|KV cache.*(failed|error|insufficient)|ACL.*[Ee]rror|HCCL.*[Ee]rror' \
  "$RUN_ROOT/runner.log" "$RUN_ROOT"/{fcfs,sjf,rhsail}.log 2>/dev/null
```

没有输出表示未发现这些严重错误。不要因为进度短暂停留而手动重启；先同时确认 runner、日志时间、NPU 活动和错误扫描。

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
  --expected-count 350

touch "$RUN_ROOT/DONE"
cat "$RUN_ROOT/final_brief.txt"
```

FCFS、SJF、RH-SAIL 都必须显示 `done=350/350 ok=100.0%`，并且存在 `DONE`、不存在错误关键词。

## 6. 结果位置

因为输出写在 Docker 挂载目录 `/x`，退出 Docker 后宿主机可以直接访问，不需要 `docker cp`：

```text
/x/outputs/<时间>-first50-rate012-fcfs-sjf-rhsail/
/x/outputs/<时间>-first50-rate003-fcfs-sjf-rhsail/
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

## 7. 从已完成数据生成详细简报

这一步只读取已经生成的 detail JSON 和 summary JSON，不会重新运行模型，也不会覆盖原有的 `final_brief.*`。两台机器的结果目录不同，因此在每台机器上分别设置自己的 `RUN_ROOT` 并运行一次：

```bash
cd /x/mfe-ascend
export RUN_ROOT=/x/outputs/<本机已经完成的 first50 结果目录>

test -f "$RUN_ROOT/DONE" && echo RUN_COMPLETE
python mfe/scripts/summarize_scheduler_runs_detailed.py "$RUN_ROOT" \
  --schedulers fcfs sjf rhsail \
  --expected-count 350 \
  --prefix detailed_brief
```

脚本要求三个策略均为 `350/350、100%`；数据缺失或不完整时会直接报错，不会生成看似完整的简报。成功后查看：

```bash
cat "$RUN_ROOT/detailed_brief.txt"
cat "$RUN_ROOT/detailed_brief.md"
```

新增文件：

```text
detailed_brief.md               完整 Markdown 简报，包含总览和 dataset 对比表
detailed_brief.txt              适合终端查看的核心指标
detailed_brief_overall.csv      每个策略一行的完整总体指标
detailed_brief_by_dataset.csv   每个 dataset × 策略一行的 service/run time
```

总体指标包括 input/output/total tokens/s、平均 run time、P99/Max service、P99/Max completion、P95 最大算子间空档、调度开销秒数及 makespan 占比，并保留 makespan、到达结束、排空、等待、ready 和 device busy。dataset 表比较 FCFS、SJF、RH-SAIL 的平均 service time 与平均 run time。

这里的 `run time` 是每个请求全部 op duration 的求和，不含排队等待和 op 间空档；并行 op 的 duration 分别计入，因此 run time 可能大于从首个 op 开始到请求完成的 service window。`P95 max gap` 的计算方式是先合并每个请求内重叠的 op 活跃区间，取该请求最大的相邻区间空档，再对全部请求的最大空档取 P95。
