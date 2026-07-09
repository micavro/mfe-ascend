# MFE Ascend

MFE Ascend 是基于 `micavro/mfe` 的昇腾迁移版本。它保留原来的 YAML DAG、多请求调度、submit/status 查询和 benchmark 输出，运行后端从默认 CUDA/vLLM 调整为面向 openEuler + Huawei Ascend NPU + vLLM Ascend。

## 核心变化

- 项目名改为 `mfe-ascend`，源码整理到标准 Python 包 `mfe/`。
- 默认推理后端为 Ascend，可通过 `MFE_ACCELERATOR=ascend|cuda` 切换。
- 调度器运行时探测可见设备，不写死卡数。
- Worker 在导入 vLLM 前设置 `ASCEND_RT_VISIBLE_DEVICES`、`NPU_VISIBLE_DEVICES`、`VLLM_TARGET_DEVICE=npu`、`VLLM_WORKER_MULTIPROC_METHOD=spawn`。
- 默认不安装或覆盖容器内的 `torch`、`torch-npu`、`vllm`、`vllm-ascend`；旧版组合记录在 `constraints/`。

## 统一运行方式

推荐使用 `deploy/run_unified.sh` 作为 A800/CUDA 和 Ascend/NPU 的统一入口。profile 只决定后端类型；卡数和卡号由运行参数指定。

Ascend 8 卡环境示例：

```bash
bash deploy/run_unified.sh company-ascend \
  --mode check \
  --model-path /data/mfe/models/Qwen3-0.6B \
  --device-ids 0,1,2,3,4,5,6,7 \
  --expected-device-count 8 \
  --offline

bash deploy/run_unified.sh company-ascend \
  --model-path /data/mfe/models/Qwen3-0.6B \
  --device-ids 0,1,2,3,4,5,6,7 \
  --expected-device-count 8 \
  --dataset gsm8k \
  --yaml adv_reason_3.yaml \
  --num 20 \
  --offline
```

A800 10 卡环境示例：

```bash
bash deploy/run_unified.sh lab-a800 \
  --model-path /data/mfe/models/Qwen3-0.6B \
  --device-ids 0,1,2,3,4,5,6,7,8,9 \
  --expected-device-count 10 \
  --dataset gsm8k \
  --yaml adv_reason_3.yaml \
  --num 20 \
  --offline
```

在 Ascend Docker 里，`--device-ids` 只选择容器内 MFE 可见的逻辑设备；宿主机上的 `/dev/davinci*` 仍然要先通过 Docker 参数暴露给容器。真实推理前建议先跑：

```bash
bash deploy/run_unified.sh company-ascend \
  --mode smoke \
  --model-path /data/mfe/models/Qwen3-0.6B \
  --device-ids 0 \
  --offline
```

无真实卡或只验证调度流程时：

```bash
bash deploy/run_unified.sh custom \
  --accelerator ascend \
  --mode test-worker \
  --dataset gsm8k \
  --yaml adv_reason_3.yaml \
  --num 1 \
  --skip-install
```

## 调度方案

当前代码里有两个可运行的调度方案，调度单位都是 `(query, operator)`，一个 `Operator` 仍对应一次 vLLM 调用；派发仍是单 query：`ExecuteInfo(query_ids=[uid])`，没有启用 query-batch。

1. `eager`：默认方案。调度器每轮收集 ready `(query, operator)`，按 ready list 顺序把第一个任务派给空闲 worker。它是当前 FCFS/ready-task baseline，适合做稳定基线。
2. `sailp`：SAI-LP/SAIL 风格的 admission-time workflow scheduler。提交 query 时根据 DAG、worker 数、可选 `reuse_from`/`reuse_group`/`eligible_devices`/`sailp` cost hints 生成 `schedule_plan`，运行时优先按计划 worker/timeline 派发 ready op。

Baseline 实验入口支持显式 `fcfs`、`sjf` 和 `sailp`：

```bash
python -m mfe.scripts.experiment_baselines \
  --questions-file data/experiments/mixed_medium_smoke.jsonl \
  --scheduler fcfs \
  --output-length medium \
  --test-worker \
  --worker-delay 0.01
```

详见 [docs/baseline-experiments.md](docs/baseline-experiments.md)。

启用 SAI-LP：

```bash
export MFE_SCHEDULER=sailp
```

也可以在单次运行前加环境变量：

```bash
MFE_SCHEDULER=sailp bash deploy/run_unified.sh company-ascend \
  --model-path /data/mfe/models/Qwen3-0.6B \
  --device-ids 0,1,2,3,4,5,6,7 \
  --expected-device-count 8 \
  --dataset gsm8k \
  --yaml sailp_example.yaml \
  --num 20 \
  --offline
```

`sailp` 不直接接管 vLLM KV cache，也不实现远程 KV 迁移；它只是通过 placement/order 偏向状态亲和。`MFE_ENABLE_PREFIX_CACHING=1` 只在当前 vLLM/vLLM Ascend 版本支持 automatic prefix caching 时才建议打开。详细配置、YAML 元数据和限制见 [docs/sailp.md](docs/sailp.md)。

## 目录结构

```text
mfe-ascend/
├── mfe/                       # Python 包
│   ├── components/            # DAG 节点、请求、执行信息
│   ├── optimizers/            # 多请求调度器
│   ├── serve/                 # submit/status server
│   ├── workers/               # vLLM Worker / TestWorker
│   ├── scripts/               # client、benchmark、环境检查脚本
│   ├── parser.py              # YAML DAG 解析
│   ├── util.py                # Ascend/CUDA 设备抽象
│   └── config.py
├── templates/                 # YAML 工作流模板
├── data/                      # 数据目录，默认从这里读取
│   └── gsm8k/
│       └── gsm8k.parquet      # GSM8K 数据文件
├── docs/openeuler-ascend.md   # openEuler/Ascend 上机说明
├── .env.ascend.example        # Ascend 环境变量模板
└── pyproject.toml
```

## 数据目录

MFE 默认把项目根目录下的 `data/` 当作数据根目录，也就是：

```text
mfe-ascend/data/
```

不同数据集放在各自子目录里，文件名与数据集名一致：

```text
mfe-ascend/data/gsm8k/gsm8k.parquet
mfe-ascend/data/drop/drop.parquet
mfe-ascend/data/hotpotqa/hotpotqa.parquet
mfe-ascend/data/math/math.parquet
```

目录名和 parquet 文件名建议全部小写。命令行里的 `--dataset GSM8k` 会被自动归一成 `gsm8k`，但实际文件路径仍应是 `data/gsm8k/gsm8k.parquet`。

如果只验证 GSM8K，小样本下载命令是：

```bash
cd /path/to/mfe-ascend
python -m mfe.scripts.download_datasets --datasets gsm8k --limit 50 --data-dir data
```

如果数据是别人提前给你的 parquet 文件，直接放到对应目录即可。例如 GSM8K：

```bash
mkdir -p data/gsm8k
cp /path/to/gsm8k.parquet data/gsm8k/gsm8k.parquet
```

运行时默认会读取 `data/`；也可以显式指定：

```bash
export MFE_DATA_DIR=$PWD/data
python -m mfe.scripts.client --dataset gsm8k --data-dir "$MFE_DATA_DIR" -n 5 --test-worker --worker-delay 0
```

vLLM 的显存使用比例可通过环境变量临时覆盖：

```bash
export MFE_GPU_MEMORY_UTILIZATION=0.95
```

如果 Llama-3.1 这类长上下文模型报 KV cache 不够，最方便的是运行时限制上下文长度：

```bash
python -m mfe.scripts.client --dataset gsm8k --max-model-len 4096 ...
```

或者用环境变量对所有 op 统一生效：

```bash
export MFE_MAX_MODEL_LEN=4096
```

也可以写进 YAML 的 op 配置里，适合固定模板参数：

```yaml
max_model_len: 4096
gpu_memory_utilization: 0.95
```

## openEuler/Ascend 环境

目标环境应先具备：

- openEuler Linux
- Ascend driver/firmware
- CANN
- Python 3.10 或 3.11
- torch-npu 与 vllm-ascend

上机后先执行：

```bash
npu-smi info
python -m mfe.scripts.check_ascend_env
```

如果你已经按 vLLM Ascend quickstart 进入 Docker，并通过 `-v` 挂载宿主机目录，下一步优先看 [docs/vllm-ascend-docker-volume.md](docs/vllm-ascend-docker-volume.md)。Ubuntu 22.04 完整部署说明见 [docs/ubuntu22-ascend-deploy.md](docs/ubuntu22-ascend-deploy.md)。openEuler 说明见 [docs/openeuler-ascend.md](docs/openeuler-ascend.md)。

## 安装

```bash
cd /path/to/mfe-ascend
python -m pip install -U pip
python -m pip install -e . --no-deps
```

如果使用官方 vLLM Ascend 容器且依赖已预装，可以先检查版本，再按需使用：

```bash
python -m pip install -e . --no-deps
```

## 运行

先按机器实际卡号设置可见设备：

```bash
source .env.ascend.example
export MFE_DEVICE_IDS=0,1
export MFE_MODEL_PATH=/data/mfe/models/Qwen3-0.6B
export MFE_DATA_DIR=$PWD/data
export MFE_OUTPUT_DIR=$PWD/data/gsm8k
```

确认模板里的模型路径指向服务器上的真实模型：

```yaml
model: "${MFE_MODEL_PATH}"
```

运行多请求测试：

```bash
python -m mfe.scripts.client --dataset gsm8k -n 20 --yaml adv_reason_3.yaml --model-path "$MFE_MODEL_PATH" --data-dir "$MFE_DATA_DIR" --offline -v
```

无 NPU 或先验证调度流程时：

```bash
python -m mfe.scripts.client --dataset gsm8k -n 5 --yaml adv_reason_3.yaml --test-worker --worker-delay 0.2 -v
```

## 注意

- 不要单独升级 `vllm` 或 `torch-npu`。Ascend 环境要求 CANN、torch、torch-npu、vllm、vllm-ascend 成套匹配。
- 当前默认依赖组合是为了贴近原项目。看到服务器实际 CANN/驱动/镜像版本后，建议再确定是否升级到更新的 vLLM Ascend 版本。
- 卡数和显存不需要预先写进代码；调度器会从 `ASCEND_RT_VISIBLE_DEVICES` 或 `torch.npu.device_count()` 获取。
