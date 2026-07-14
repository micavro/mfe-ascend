# 公司 Ascend 三策略实验现场手册

这份手册用于在公司 Ascend 机器上，以同一配置顺序运行 FCFS、SJF、RH-SAIL，并在数据不能带出公司的情况下自动生成一张可直接截图或抄录的最终简报。

## 0. 先明确运行方式

本项目不需要先启动 `vllm serve`。MFE 的每个 worker 会执行 `from vllm import LLM`，由 vLLM 自动加载 `vllm-ascend` 硬件插件，并在指定 NPU 上创建离线推理引擎。Docker 的作用是提供彼此兼容的 CANN、torch、torch-npu、vLLM 和 vLLM Ascend 环境。

如果公司只提供了一个已经启动的 OpenAI 兼容 HTTP 服务，当前 MFE 不能直接使用该服务；需要进入包含 vLLM Ascend Python 环境的容器运行本项目。

官方说明：

- [vLLM Ascend 安装和 Docker 启动](https://docs.vllm.ai/projects/ascend/en/latest/installation.html)
- [vLLM Ascend 版本兼容矩阵](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html)
- [vLLM Ascend FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs/)

不要在已经验证过的镜像中单独升级 `torch`、`torch-npu`、`vllm` 或 `vllm-ascend`。这些包和 CANN 必须按兼容矩阵成套匹配。

## 1. 出发前准备

建议带入或提前放到公司服务器的目录结构：

```text
/data/mfe/
├── mfe-ascend/
│   ├── deploy/run_company_ascend_sweep.sh
│   ├── deploy/start_company_ascend_container.sh
│   ├── mfe/scripts/summarize_scheduler_runs.py
│   ├── data/experiments_design7/mixed_medium_first200.jsonl
│   └── templates/
├── models/
│   └── MODEL_NAME/
│       ├── config.json
│       ├── tokenizer_config.json
│       └── *.safetensors
└── outputs/
```

本次默认实验口径：

| 项目 | 默认值 |
|---|---|
| 策略 | FCFS、SJF、RH-SAIL，依次运行 |
| 请求 | `mixed_medium_first200.jsonl`，1400 条 |
| 到达 | Poisson burst，batch size 1 |
| 到达率 | 0.13 req/s，可通过 `POISSON_RATE` 修改 |
| seed | 20260709 |
| max model len | 32768 |
| 每个 DAG op 最大输出 | 2048 tokens |
| vLLM memory utilization | 0.75 |
| prefix caching | 关闭 |

出发前在项目根目录运行一次本地测试：

```bash
python -m unittest discover -s tests -p 'test_summarize_scheduler_runs.py' -v
wc -l data/experiments_design7/mixed_medium_first200.jsonl
```

第二条应输出 `1400`。如果需要离线压缩代码和数据：

```bash
cd /path/containing/mfe-ascend
tar --exclude='outputs' --exclude='.git' -czf mfe-ascend-company.tgz mfe-ascend
sha256sum mfe-ascend-company.tgz > mfe-ascend-company.tgz.sha256
```

模型通常很大，优先使用公司已有的本地模型目录，不要在现场临时从 Hugging Face 下载。

## 2. 宿主机检查

先确认管理员分配给你的 NPU 编号。以下例子使用 0 到 7；如果只分配了部分卡，后续所有 `NPU_DEVICE_IDS` 和 `DEVICE_IDS` 都改为实际编号。不要停止、重置或占用其他人的卡。

```bash
date
uname -a
cat /etc/os-release
docker version
npu-smi info
ls -l /dev/davinci* /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
df -h /data
docker images | grep -E 'vllm|ascend|cann'
```

继续运行的必要条件：

1. `npu-smi info` 正常，分配的 NPU 没有陌生计算进程或异常状态。
2. 对应 `/dev/davinciN`、manager、SVM 和 HDC 设备文件存在。
3. 公司批准的 vLLM Ascend 镜像已经在本机，或者公司镜像仓库可访问。
4. 模型、1400 条 JSONL、模板和输出目录可读写，磁盘空间足够。

任一条件不满足，先找管理员处理，不要靠重装驱动或重置 NPU 解决。

## 3. 启动 Docker

优先使用公司已经验证过的镜像标签。官方文档中的最新示例会变化，不要仅因为标签更新就替换公司的兼容环境。

宿主机执行：

```bash
cd /data/mfe/mfe-ascend

export IMAGE='<公司批准的 vllm-ascend 镜像完整名称>'
export HOST_WORK=/data/mfe
export NPU_DEVICE_IDS=0,1,2,3,4,5,6,7
export CONTAINER_NAME=mfe-ascend-company
export SHM_SIZE=16g

bash deploy/start_company_ascend_container.sh
docker ps --filter name="$CONTAINER_NAME"
docker exec -it "$CONTAINER_NAME" bash
```

启动脚本采用官方 Docker quickstart 的设备和驱动挂载方式，只把指定的 `/dev/davinciN` 暴露给容器，并将 `/data/mfe` 挂载为容器内相同用途的工作目录。容器以 detached 模式运行，SSH 断开不会自动销毁容器。

如果公司不允许使用辅助脚本，可让管理员查看该脚本后用等价的 `docker run` 命令启动。不要为了省事使用 `--privileged`，也不要把未分配的 NPU 设备传入容器。

## 4. 容器内环境检查

进入容器后执行：

```bash
cd /data/mfe/mfe-ascend

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi

npu-smi info
python - <<'PY'
import importlib.metadata as m

for package in ("torch", "torch-npu", "vllm", "vllm-ascend", "transformers"):
    try:
        print(f"{package}={m.version(package)}")
    except Exception as exc:
        print(f"{package}=ERROR: {exc}")

import torch
import torch_npu
import vllm
import vllm_ascend

print("torch.npu.is_available=", torch.npu.is_available())
print("torch.npu.device_count=", torch.npu.device_count())
PY
```

必须满足：

- 五个包都能显示版本，四个 import 不报错。
- `torch.npu.is_available=True`。
- `torch.npu.device_count()` 与容器映射的卡数一致。
- `vllm` 与 `vllm-ascend` 版本满足官方兼容矩阵。

只安装项目本身时使用：

```bash
python -m pip install -e . --no-deps
```

不要省略 `--no-deps`，否则 pip 可能覆盖镜像里匹配好的运行栈。直接从项目根目录运行时通常不安装也能导入 `mfe`。

## 5. 模型、数据和完整体检

先设置一次现场参数：

```bash
export MODEL_PATH=/data/mfe/models/MODEL_NAME
export QUESTIONS_FILE=/data/mfe/mfe-ascend/data/experiments_design7/mixed_medium_first200.jsonl
export DEVICE_IDS=0,1,2,3,4,5,6,7
export EXPECTED_DEVICE_COUNT=8

test -f "$MODEL_PATH/config.json"
test -f "$QUESTIONS_FILE"
grep -cve '^[[:space:]]*$' "$QUESTIONS_FILE"
du -sh "$MODEL_PATH"
df -h /data/mfe

bash deploy/run_unified.sh company-ascend \
  --mode check \
  --model-path "$MODEL_PATH" \
  --device-ids "$DEVICE_IDS" \
  --expected-device-count "$EXPECTED_DEVICE_COUNT" \
  --questions-file "$QUESTIONS_FILE" \
  --offline \
  --skip-install
```

检查应以退出码 0 结束。请求数必须是 `1400`。

## 6. 单卡 vLLM Ascend smoke test

全量实验前先只用一张已分配的卡验证模型确实能由 vLLM Ascend 加载：

```bash
bash deploy/run_unified.sh company-ascend \
  --mode smoke \
  --model-path "$MODEL_PATH" \
  --device-ids 0 \
  --expected-device-count 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.75 \
  --offline \
  --skip-install
```

成功标准是最终打印一小段模型回答并以退出码 0 结束。第一次加载可能需要几分钟。以下情况不要继续全量运行：

- `Out of memory`、`OOM` 或 HBM 分配失败。
- `context length`、`maximum context` 或 KV cache 容量错误。
- `torch_npu`、ACL、HCCL、CANN 动态库错误。
- `Failed to infer device type` 或 `libatb.so` 缺失。
- Python traceback、进程退出或 NPU 健康状态异常。

## 7. 后台运行三种策略

建议为每次实验创建全新的输出目录。容器内执行：

```bash
export MODEL_PATH=/data/mfe/models/MODEL_NAME
export QUESTIONS_FILE=/data/mfe/mfe-ascend/data/experiments_design7/mixed_medium_first200.jsonl
export DEVICE_IDS=0,1,2,3,4,5,6,7
export EXPECTED_DEVICE_COUNT=8
export EXPECTED_REQUESTS=1400
export POISSON_RATE=0.13
export ARRIVAL_SEED=20260709
export ARRIVAL_BATCH_SIZE=1
export MAX_MODEL_LEN=32768
export OUTPUT_MAX_TOKENS=2048
export GPU_MEMORY_UTILIZATION=0.75
export SCHEDULERS='fcfs sjf rhsail'
export OUTPUT_ROOT="/data/mfe/outputs/$(date +%Y%m%d-%H%M%S)-poisson013-fcfs-sjf-rhsail"

nohup bash deploy/run_company_ascend_sweep.sh \
  > /data/mfe/last_company_sweep_launcher.log 2>&1 &
echo $!
```

脚本会完成以下工作：

1. 检查模型、1400 条请求、包、NPU 数量和同构性。
2. 固定相同到达序列，依次运行 FCFS、SJF、RH-SAIL。
3. 每个策略结束后验证 `1400/1400` 和 `success_rate=100%`。
4. 扫描 OOM、traceback、CUDA、context、KV cache、ACL、HCCL 错误。
5. 自动生成 `final_brief.md`、`final_brief.txt`、`final_brief.csv`，最后创建 `DONE`。

## 8. 运行中怎么监控

另开一个容器终端：

```bash
docker exec -it mfe-ascend-company bash
cd /data/mfe/mfe-ascend

cat "$OUTPUT_ROOT/runner.pid"
ps -fp "$(cat "$OUTPUT_ROOT/runner.pid")"
tail -n 100 -f "$OUTPUT_ROOT/runner.log"
```

查看完成到哪一个策略：

```bash
find "$OUTPUT_ROOT" -maxdepth 2 -type f \
  \( -name 'brief_summary.txt' -o -name '*_summary.json' -o -name DONE \) -print

grep -E 'START scheduler=|DONE scheduler=|ALL DONE' "$OUTPUT_ROOT/runner.log"
```

查看 NPU：

```bash
watch -n 5 npu-smi info
```

只做错误扫描：

```bash
grep -Eain \
  'out of memory|OOM|Traceback|CUDA error|context length|maximum context|KV cache|ACL.*[Ee]rror|HCCL.*[Ee]rror' \
  "$OUTPUT_ROOT"/fcfs.log "$OUTPUT_ROOT"/sjf.log "$OUTPUT_ROOT"/rhsail.log
```

正常现象：

- 首次模型加载较慢，随后日志继续更新。
- 分配的 NPU 有计算和 HBM 占用，未映射的 NPU 不受影响。
- 策略按 FCFS、SJF、RH-SAIL 顺序运行，同一时刻只有一个策略实验。
- 每个策略目录最终有 detail JSON、summary JSON、`brief_summary.csv/txt`。

异常现象：

- `runner.pid` 不存在或进程消失，但没有 `DONE`。
- 日志长时间不更新且 NPU 持续 0 利用率。
- 请求数不是 1400、成功率低于 100%，或出现上面的错误关键词。
- 容器看到的设备数与 `EXPECTED_DEVICE_COUNT` 不一致。

发现异常时先保存日志并确认 PID。不要使用模糊的 `pkill python`，也不要操作未分配的 NPU 或其他容器。

## 9. 完成后的验收和自动简报

```bash
test -f "$OUTPUT_ROOT/DONE" && echo 'ALL COMPLETE'
test ! -s "$OUTPUT_ROOT/error_scan.txt" && echo 'NO FATAL KEYWORDS'
cat "$OUTPUT_ROOT/final_brief.txt"
```

终端会得到一张很短的表：

```text
MFE_FINAL_BRIEF_START
FCFS done=1400/1400 ok=100.0% makespan=... arrival_end=... drain=... tokens/s=... wait=... service=... completion=... ready=... device_busy=...
SJF ...
RH-SAIL ...
MFE_FINAL_BRIEF_END
```

指标定义：

- `Makespan`：从第一条请求到达到最后一条请求完成。
- `Arrival end`：最后一条请求相对第一条请求的到达时间。
- `Drain`：到达结束后把系统内剩余请求排空所需时间，即 makespan 减 arrival end。
- `Tokens/s`：输入与输出 token 总数除以 makespan。
- `Avg wait`：请求平均等待时间。
- `Avg service`：请求 DAG 各 op 的执行区间总和的平均值。
- `Avg completion`：从请求到达到请求完成的平均时间，即平均 latency。
- `Ready avg/peak`：调度器 ready 队列平均长度和峰值。
- `Device busy`：各 worker 调度层忙碌占比的平均值，不等同于 `npu-smi` 的瞬时硬件利用率。

公司数据不能带出时，只需保留或拍下 `MFE_FINAL_BRIEF_START/END` 之间的表格，并额外记录镜像名、模型路径、NPU 型号、卡数、git commit 和 Poisson rate。

如果简报文件被误删，可重新生成，不会重新跑实验：

```bash
python -m mfe.scripts.summarize_scheduler_runs "$OUTPUT_ROOT" \
  --schedulers fcfs sjf rhsail \
  --expected-count 1400
```

## 10. 中断后的恢复

每个策略只在完整结束时写 summary JSON，因此不能从单个策略的中间请求续跑。可以保留已经完成的策略，重新运行未完成策略：

```bash
export RESUME=1
nohup bash deploy/run_company_ascend_sweep.sh \
  > /data/mfe/last_company_sweep_resume.log 2>&1 &
```

脚本只会跳过存在唯一 summary、且达到 `1400/1400`、成功率 100% 的策略。若失败策略目录里已有不完整 summary，先把整个失败目录改名留档，再恢复：

```bash
mv "$OUTPUT_ROOT/sjf" "$OUTPUT_ROOT/sjf.failed.$(date +%Y%m%d-%H%M%S)"
rm -f "$OUTPUT_ROOT/sjf.log"
```

不要删除已经完成策略的 JSON；不要为了恢复实验清理整机 Python 进程。

## 11. 收尾

确认简报已记录后，在宿主机停止本次专用容器：

```bash
docker stop mfe-ascend-company
docker rm mfe-ascend-company
```

这只停止本次创建的容器，不会操作宿主机上的其他容器或 NPU 进程。
