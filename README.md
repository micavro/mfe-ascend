# MFE Ascend

MFE Ascend 是基于 `micavro/mfe` 的昇腾迁移版本。它保留原来的 YAML DAG、多请求调度、submit/status 查询和 benchmark 输出，运行后端从默认 CUDA/vLLM 调整为面向 openEuler + Huawei Ascend NPU + vLLM Ascend。

## 核心变化

- 项目名改为 `mfe-ascend`，源码整理到标准 Python 包 `mfe/`。
- 默认推理后端为 Ascend，可通过 `MFE_ACCELERATOR=ascend|cuda|auto` 切换。
- 调度器运行时探测可见设备，不写死卡数。
- Worker 在导入 vLLM 前设置 `ASCEND_RT_VISIBLE_DEVICES`、`NPU_VISIBLE_DEVICES`、`VLLM_TARGET_DEVICE=npu`。
- 依赖先按原项目 `vllm==0.9.0` 对齐到 `vllm-ascend==0.9.0rc2`、`torch==2.5.1`、`torch-npu==2.5.1`。

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
├── docs/openeuler-ascend.md   # openEuler/Ascend 上机说明
├── .env.ascend.example        # Ascend 环境变量模板
└── pyproject.toml
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

Ubuntu 22.04 服务器部署和试运行优先看 [docs/ubuntu22-ascend-deploy.md](docs/ubuntu22-ascend-deploy.md)。openEuler 说明见 [docs/openeuler-ascend.md](docs/openeuler-ascend.md)。

## 安装

```bash
cd /path/to/mfe-ascend
python -m pip install -U pip
python -m pip install -e .
```

如果使用官方 vLLM Ascend 容器且依赖已预装，可以先检查版本，再按需使用：

```bash
python -m pip install -e . --no-deps
```

## 运行

先按机器实际卡号设置可见设备：

```bash
source .env.ascend.example
export ASCEND_RT_VISIBLE_DEVICES=0,1
export NPU_VISIBLE_DEVICES=0,1
```

确认模板里的模型路径指向服务器上的真实模型：

```yaml
model: /data/models/Llama-3.1-8B-Instruct
```

运行多请求测试：

```bash
python -m mfe.scripts.client --dataset gsm8k -n 20 --yaml adv_reason_3.yaml --send-interval 0.0 -v
```

无 NPU 或先验证调度流程时：

```bash
python -m mfe.scripts.client --dataset gsm8k -n 5 --yaml adv_reason_3.yaml --test-worker --worker-delay 0.2 -v
```

## 注意

- 不要单独升级 `vllm` 或 `torch-npu`。Ascend 环境要求 CANN、torch、torch-npu、vllm、vllm-ascend 成套匹配。
- 当前默认依赖组合是为了贴近原项目。看到服务器实际 CANN/驱动/镜像版本后，建议再确定是否升级到更新的 vLLM Ascend 版本。
- 卡数和显存不需要预先写进代码；调度器会从 `ASCEND_RT_VISIBLE_DEVICES` 或 `torch.npu.device_count()` 获取。
