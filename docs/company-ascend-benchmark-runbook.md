# 已有 vLLM Docker 的五卡双机实验操作单

现状：两台服务器上已经各有一个配置好 Python、vLLM Ascend、并映射了 8 张 NPU 的运行中 Docker。我们不重新搭 Docker，直接把 MFE 项目复制进现有容器，每台只选择其中 5 张卡运行。

- 服务器 A：`0.12 req/s`，依次运行 FCFS、SJF、RH-SAIL。
- 服务器 B：`0.15 req/s`，依次运行 FCFS、SJF、RH-SAIL。
- 默认使用容器内 NPU `0,1,2,3,4`。

重要：现有容器可以是 vLLM 环境容器，但选择的五张卡上不能同时运行 `vllm serve` 或其他推理任务，否则会争抢 HBM。启动脚本会检查常见 vLLM API 服务进程，发现后拒绝启动，不会自动停止任何进程。

## 1. 下载项目 ZIP 并上传

在自己的 Windows PowerShell 下载默认 `main`：

```powershell
$zip = "$HOME\Downloads\mfe-ascend.zip"
Invoke-WebRequest `
  -Uri "https://github.com/micavro/mfe-ascend/archive/refs/heads/main.zip" `
  -OutFile $zip
python -m zipfile -t $zip
```

最后一行必须输出 `Done testing`。ZIP 约 33 MB，GitHub 生成和下载可能需要约两分钟。

通过 SSH 上传到两台服务器：

```powershell
scp -P 22 $zip USER@SERVER_A:/data/mfe/
scp -P 22 $zip USER@SERVER_B:/data/mfe/
```

## 2. 在宿主机解压项目

两台服务器都执行：

```bash
mkdir -p /data/mfe
cd /data/mfe

if [ -d mfe-ascend ]; then
  mv mfe-ascend "mfe-ascend.backup.$(date +%Y%m%d-%H%M%S)"
fi

python -m zipfile -e mfe-ascend.zip .
mv mfe-ascend-main mfe-ascend
cd mfe-ascend

test -f deploy/launch_in_existing_vllm_container.sh && echo SCRIPT_OK
grep -cve '^[[:space:]]*$' data/experiments_design7/mixed_medium_first200.jsonl
```

请求数必须是 `1400`。项目 ZIP 已包含实验数据和启动脚本，不包含模型权重。

## 3. 确认现有 8 卡容器可用

在宿主机查看容器名称：

```bash
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
```

下面假设容器名是 `vllm-ascend`。先检查 8 张卡、Python 环境、模型路径和活动服务：

```bash
export CONTAINER_NAME=vllm-ascend

docker exec "$CONTAINER_NAME" npu-smi info
docker exec "$CONTAINER_NAME" python -c "import torch,torch_npu,vllm,vllm_ascend; print('npu_count=',torch.npu.device_count()); print('vllm=',vllm.__version__)"
docker exec "$CONTAINER_NAME" sh -lc "ps -eo pid,args | grep -E '[v]llm[[:space:]]+serve|vllm\.entrypoints\.openai\.api_server' || true"
docker exec "$CONTAINER_NAME" test -f /容器内模型路径/config.json && echo MODEL_OK
```

要求：

- `npu_count=8`，说明原容器确实映射了八张卡；
- Python import 没有 traceback；
- 模型 `config.json` 存在；
- 五张待用卡空闲；
- 服务进程检查没有输出。如果有输出，先由容器负责人处理，不能直接叠加实验。

不需要在容器里 `pip install`。启动脚本会自动执行：

```text
docker cp 宿主机项目/. 容器:/workspace/mfe-ascend/
```

## 4. 直接启动脚本

两个命令都在宿主机的 `/data/mfe/mfe-ascend` 下执行。将 `vllm-ascend` 改成实际容器名，将 `/容器内模型路径` 改成该容器当前使用的模型目录。

共同参数已经固定为：5 张卡 `0,1,2,3,4`、1400 请求、Poisson batch size 1、seed 20260709、max model len 32768、每个 op 最大输出 2048、memory utilization 0.75、prefix caching 关闭。

### 服务器 A：0.12 req/s

```bash
cd /data/mfe/mfe-ascend
CONTAINER_NAME=vllm-ascend \
MODEL_PATH=/容器内模型路径 \
POISSON_RATE=0.12 \
DEVICE_IDS=0,1,2,3,4 \
bash deploy/launch_in_existing_vllm_container.sh | tee /data/mfe/launch-rate012.txt
```

### 服务器 B：0.15 req/s

```bash
cd /data/mfe/mfe-ascend
CONTAINER_NAME=vllm-ascend \
MODEL_PATH=/容器内模型路径 \
POISSON_RATE=0.15 \
DEVICE_IDS=0,1,2,3,4 \
bash deploy/launch_in_existing_vllm_container.sh | tee /data/mfe/launch-rate015.txt
```

脚本成功时会输出：

```text
STARTED container=... rate=... devices=0,1,2,3,4
OUTPUT_ROOT=/workspace/mfe-results/...
```

把 `OUTPUT_ROOT` 记下来。它也保存在宿主机的 `launch-rate012.txt` 或 `launch-rate015.txt` 中。

## 5. 检查是否正常运行和完成

从启动输出中恢复结果目录：

```bash
# 服务器 A
OUT="$(sed -n 's/^OUTPUT_ROOT=//p' /data/mfe/launch-rate012.txt | tail -n 1)"

# 服务器 B 使用：
# OUT="$(sed -n 's/^OUTPUT_ROOT=//p' /data/mfe/launch-rate015.txt | tail -n 1)"

echo "$OUT"
```

实时日志、策略进度和 NPU 状态：

```bash
docker exec "$CONTAINER_NAME" tail -n 100 -f "$OUT/runner.log"
docker exec "$CONTAINER_NAME" grep -E 'START scheduler=|DONE scheduler=|ALL DONE|FAILED' "$OUT/runner.log"
docker exec "$CONTAINER_NAME" npu-smi info
```

检查错误：

```bash
docker exec "$CONTAINER_NAME" sh -lc "grep -Eain 'out of memory|OOM|Traceback|CUDA error|context length|maximum context|KV cache|ACL.*[Ee]rror|HCCL.*[Ee]rror' '$OUT'/*.log || true"
```

全部完成必须同时满足：

```bash
docker exec "$CONTAINER_NAME" test -f "$OUT/DONE" && echo ALL_COMPLETE
docker exec "$CONTAINER_NAME" test ! -f "$OUT/FAILED" && echo NO_FAILED_MARKER
docker exec "$CONTAINER_NAME" test ! -s "$OUT/error_scan.txt" && echo NO_FATAL_KEYWORDS
docker exec "$CONTAINER_NAME" cat "$OUT/final_brief.txt"
```

`final_brief.txt` 中 FCFS、SJF、RH-SAIL 都必须显示 `done=1400/1400 ok=100.0%`。

## 6. 从容器收集结果

结果最初位于现有容器内部：

```text
/workspace/mfe-results/<时间>-rate012-fcfs-sjf-rhsail
/workspace/mfe-results/<时间>-rate015-fcfs-sjf-rhsail
```

复制到服务器宿主机持久保存：

```bash
mkdir -p /data/mfe/results
docker cp "$CONTAINER_NAME:$OUT" /data/mfe/results/
```

宿主机结果目录包含：

```text
run_config.txt
preflight.json
runner.log
fcfs/、sjf/、rhsail/ 的 detail JSON 和 summary JSON
final_brief.txt
final_brief.md
final_brief.csv
error_scan.txt
DONE
```

无法带出公司数据时，至少拍下：

```bash
cat /data/mfe/results/$(basename "$OUT")/run_config.txt
cat /data/mfe/results/$(basename "$OUT")/final_brief.txt
```
