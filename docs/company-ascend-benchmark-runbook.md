# 公司 Ascend 五卡实验操作单

目标：两台服务器各用 5 张 Ascend 卡，服务器 A 顺序运行 `0.12 req/s` 下的 FCFS、SJF、RH-SAIL，服务器 B 顺序运行 `0.15 req/s` 下的三个策略。服务器没有外网，Python 和 vLLM Ascend 环境已经配好。

本文默认：

```text
项目目录：/data/mfe/mfe-ascend
模型目录：/data/mfe/models/MODEL_NAME
输出目录：/data/mfe/outputs
容器名称：mfe-ascend-company
五张卡：0,1,2,3,4
```

如果管理员分配的不是 0 到 4，把全文中的卡号换成实际五张卡。不要操作其他人的 NPU。

## 1. 下载 ZIP 并通过 SSH 上传

在自己的 Windows 电脑上打开 PowerShell，下载当前实验分支的完整 ZIP：

```powershell
$url = "https://github.com/micavro/mfe-ascend/archive/refs/heads/codex/rh-sail-scheduler.zip"
$zip = "$HOME\Downloads\mfe-ascend.zip"
Invoke-WebRequest -Uri $url -OutFile $zip
Get-FileHash $zip -Algorithm SHA256
```

通过 SSH 协议上传到两台服务器。替换用户名、地址和 SSH 端口：

```powershell
scp -P 22 $zip USER@SERVER_A:/data/mfe/
scp -P 22 $zip USER@SERVER_B:/data/mfe/
```

如果模型不在服务器上，也要提前用 `scp` 上传模型目录或模型压缩包。项目 ZIP 不包含模型权重。

服务器无网络时还必须保证 vLLM Ascend Docker 镜像已经存在：

```bash
docker images | grep -E 'vllm|ascend'
```

如果没有镜像，请使用公司已经验证的镜像 tar，由管理员或可联网的兼容 Linux 机器执行 `docker save`，再通过 `scp` 上传，服务器执行：

```bash
docker load -i /data/mfe/vllm-ascend-image.tar
```

不要在现场随意更换 `torch`、`torch-npu`、`vllm` 或 `vllm-ascend` 版本。

## 2. 解压项目和数据

以下操作在两台服务器上都执行：

```bash
mkdir -p /data/mfe
cd /data/mfe

# 如果旧目录存在，先改名保留，不直接删除。
if [ -d mfe-ascend ]; then
  mv mfe-ascend "mfe-ascend.backup.$(date +%Y%m%d-%H%M%S)"
fi

python -m zipfile -e mfe-ascend.zip .
EXTRACTED_DIR="$(find . -maxdepth 1 -type d -name 'mfe-ascend-*' | head -n 1)"
mv "$EXTRACTED_DIR" mfe-ascend
cd mfe-ascend
```

GitHub ZIP 已经包含实验 JSONL，不需要另外下载数据。检查项目和 1400 条请求：

```bash
test -f deploy/run_company_ascend_sweep.sh && echo PROJECT_OK
test -f data/experiments_design7/mixed_medium_first200.jsonl && echo DATA_OK
grep -cve '^[[:space:]]*$' data/experiments_design7/mixed_medium_first200.jsonl
test -f /data/mfe/models/MODEL_NAME/config.json && echo MODEL_OK
```

请求数必须输出 `1400`。如果收到的是额外数据压缩包，再按格式解压到项目目录：

```bash
tar -xzf /data/mfe/data.tar.gz -C /data/mfe/mfe-ascend
# 或：python -m zipfile -e /data/mfe/data.zip /data/mfe/mfe-ascend
```

## 3. 启动五卡 Docker

先在宿主机检查五张卡和镜像：

```bash
npu-smi info
ls -l /dev/davinci{0,1,2,3,4} /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc
docker images | grep -E 'vllm|ascend'
```

设置公司实际镜像名并启动容器：

```bash
cd /data/mfe/mfe-ascend
export IMAGE='<docker images 中公司已经验证的 vllm-ascend 镜像名:标签>'
export HOST_WORK=/data/mfe
export NPU_DEVICE_IDS=0,1,2,3,4
export CONTAINER_NAME=mfe-ascend-company

bash deploy/start_company_ascend_container.sh
docker ps --filter name=mfe-ascend-company
```

确认容器内 Python、vLLM Ascend 和五张卡正常：

```bash
docker exec mfe-ascend-company npu-smi info
docker exec mfe-ascend-company python -c "import torch,torch_npu,vllm,vllm_ascend; print('npu_available=',torch.npu.is_available()); print('npu_count=',torch.npu.device_count()); print('vllm=',vllm.__version__)"
```

必须看到：

- `npu_available=True`；
- `npu_count=5`；
- `npu-smi info` 能看到指定五张卡；
- Python import 没有 traceback。

## 4. 用一条完整命令开跑

两台机器除 `POISSON_RATE` 和 `OUTPUT_ROOT` 外，其他参数完全相同：

| 参数 | 值 |
|---|---|
| 策略顺序 | `fcfs sjf rhsail` |
| 请求文件 | `mixed_medium_first200.jsonl` |
| 请求数 | 1400 |
| NPU | `0,1,2,3,4`，共 5 张 |
| 到达模式 | Poisson burst，batch size 1 |
| 到达 seed | 20260709 |
| max model len | 32768 |
| 每个 DAG op 最大输出 | 2048 tokens |
| vLLM memory utilization | 0.75 |
| prefix caching | 关闭 |
| 离线模式 | 开启 |

先把命令中的 `MODEL_NAME` 换成真实模型目录名。

### 服务器 A：0.12 req/s

在宿主机直接执行这一整段：

```bash
OUT012="/data/mfe/outputs/$(date +%Y%m%d-%H%M%S)-rate012-fcfs-sjf-rhsail"
docker exec -d \
  -e MODEL_PATH=/data/mfe/models/MODEL_NAME \
  -e QUESTIONS_FILE=/data/mfe/mfe-ascend/data/experiments_design7/mixed_medium_first200.jsonl \
  -e DEVICE_IDS=0,1,2,3,4 \
  -e EXPECTED_DEVICE_COUNT=5 \
  -e EXPECTED_REQUESTS=1400 \
  -e POISSON_RATE=0.12 \
  -e ARRIVAL_SEED=20260709 \
  -e ARRIVAL_BATCH_SIZE=1 \
  -e MAX_MODEL_LEN=32768 \
  -e OUTPUT_MAX_TOKENS=2048 \
  -e GPU_MEMORY_UTILIZATION=0.75 \
  -e "SCHEDULERS=fcfs sjf rhsail" \
  -e MFE_ENABLE_PREFIX_CACHING=0 \
  -e OUTPUT_ROOT="$OUT012" \
  mfe-ascend-company \
  bash -lc 'cd /data/mfe/mfe-ascend && bash deploy/run_company_ascend_sweep.sh'
echo "OUTPUT_ROOT=$OUT012"
```

### 服务器 B：0.15 req/s

```bash
OUT015="/data/mfe/outputs/$(date +%Y%m%d-%H%M%S)-rate015-fcfs-sjf-rhsail"
docker exec -d \
  -e MODEL_PATH=/data/mfe/models/MODEL_NAME \
  -e QUESTIONS_FILE=/data/mfe/mfe-ascend/data/experiments_design7/mixed_medium_first200.jsonl \
  -e DEVICE_IDS=0,1,2,3,4 \
  -e EXPECTED_DEVICE_COUNT=5 \
  -e EXPECTED_REQUESTS=1400 \
  -e POISSON_RATE=0.15 \
  -e ARRIVAL_SEED=20260709 \
  -e ARRIVAL_BATCH_SIZE=1 \
  -e MAX_MODEL_LEN=32768 \
  -e OUTPUT_MAX_TOKENS=2048 \
  -e GPU_MEMORY_UTILIZATION=0.75 \
  -e "SCHEDULERS=fcfs sjf rhsail" \
  -e MFE_ENABLE_PREFIX_CACHING=0 \
  -e OUTPUT_ROOT="$OUT015" \
  mfe-ascend-company \
  bash -lc 'cd /data/mfe/mfe-ascend && bash deploy/run_company_ascend_sweep.sh'
echo "OUTPUT_ROOT=$OUT015"
```

脚本启动后会先做环境、模型、数据、五卡和参数检查。实际使用的参数保存在 `$OUTPUT_ROOT/run_config.txt`，立即核对：

```bash
docker exec mfe-ascend-company cat "$OUT012/run_config.txt"   # 服务器 A
docker exec mfe-ascend-company cat "$OUT015/run_config.txt"   # 服务器 B
```

## 5. 检查是否正常运行和完成

以下命令中的 `$OUT`，在服务器 A 设为 `$OUT012`，服务器 B 设为 `$OUT015`。

如果重新登录后变量丢失，按机器恢复最新输出目录：

```bash
# 服务器 A
OUT="$(docker exec mfe-ascend-company sh -lc "ls -dt /data/mfe/outputs/*-rate012-* | head -n 1")"

# 服务器 B 改用这一行
# OUT="$(docker exec mfe-ascend-company sh -lc "ls -dt /data/mfe/outputs/*-rate015-* | head -n 1")"

echo "$OUT"
```

查看实时日志：

```bash
docker exec mfe-ascend-company tail -n 100 -f "$OUT/runner.log"
```

查看当前策略和已完成策略：

```bash
docker exec mfe-ascend-company grep -E 'START scheduler=|DONE scheduler=|ALL DONE|FAILED' "$OUT/runner.log"
docker exec mfe-ascend-company sh -lc "find '$OUT' -maxdepth 2 -name '*_summary.json' -o -name DONE -o -name FAILED"
```

查看进程和 NPU：

```bash
docker exec mfe-ascend-company sh -lc "ps -fp \$(cat '$OUT/runner.pid')"
docker exec mfe-ascend-company npu-smi info
```

扫描关键错误：

```bash
docker exec mfe-ascend-company sh -lc "grep -Eain 'out of memory|OOM|Traceback|CUDA error|context length|maximum context|KV cache|ACL.*[Ee]rror|HCCL.*[Ee]rror' '$OUT'/*.log || true"
```

正常运行应满足：日志持续更新；依次出现 FCFS、SJF、RH-SAIL；五张卡有 HBM/计算占用；没有上述错误。

全部完成应满足：

```bash
docker exec mfe-ascend-company test -f "$OUT/DONE" && echo ALL_COMPLETE
docker exec mfe-ascend-company test ! -f "$OUT/FAILED" && echo NO_FAILED_MARKER
docker exec mfe-ascend-company test ! -s "$OUT/error_scan.txt" && echo NO_FATAL_KEYWORDS
docker exec mfe-ascend-company cat "$OUT/final_brief.txt"
```

最终简报必须显示三个策略均为 `done=1400/1400 ok=100.0%`。

## 6. 数据收集位置

所有结果都在启动时设置的 `$OUTPUT_ROOT`，即：

```text
/data/mfe/outputs/<时间>-rate012-fcfs-sjf-rhsail/   # 服务器 A
/data/mfe/outputs/<时间>-rate015-fcfs-sjf-rhsail/   # 服务器 B
```

主要文件：

```text
run_config.txt       实际运行参数
preflight.json       Python、包和五卡环境检查
runner.log           总日志
fcfs.log/sjf.log/rhsail.log
fcfs/                FCFS detail JSON、summary JSON、brief_summary
sjf/                 SJF detail JSON、summary JSON、brief_summary
rhsail/              RH-SAIL detail JSON、summary JSON、brief_summary
final_brief.txt      最简终端结果
final_brief.md       Markdown 表格
final_brief.csv      可继续分析的汇总表
error_scan.txt       应为空
DONE                 全部成功标记
```

无法把公司数据带出时，至少执行并拍下：

```bash
docker exec mfe-ascend-company cat "$OUT/final_brief.txt"
docker exec mfe-ascend-company cat "$OUT/run_config.txt"
```

`final_brief.txt` 包含每个策略的 makespan、到达结束、排空时间、tokens/s、平均等待、平均 service、平均完成时间、ready avg/peak 和 device busy。
