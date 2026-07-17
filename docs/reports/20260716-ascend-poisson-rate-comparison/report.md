# Ascend 910B4 first50 泊松到达率与调度策略完整报告

## 1. 技术摘要

- **`0.03 req/s` 已接近饱和但没有形成大规模 ready 堆积，`0.12 req/s` 则明显持续过载。** `0.03` 的排空尾部仍占到达窗口 2.0%--8.2%，不能严格称为完全不拥堵；到达率提高 4 倍后，total tokens/s 只增加 5.8%--7.9%，但排空时间扩大 9.1--35.4 倍。
- **SJF 的观测平均性能最好，代价是严重的长请求 service starvation。** 它在两档负载下都有最短 makespan、最高 total tokens/s 和最低平均完成时间；但在 `0.12` 下 P99/Max service 达 2898.7/7331.2 秒。由于 SJF 实际处理的总 token 更少，不能把全部 makespan 优势归因于调度。
- **FCFS 是最强的全局尾部基线。** 它持续取得最低 P99/Max completion 和最低 P95 max gap，说明自然 arrival order 对当前长度差异大的 DAG workload 提供了很强的 aging。
- **RH-SAIL 的主要价值是过载下的有界 admitted-ready 压力和长 DAG 连续性。** `0.12` 下 ready avg/peak 仅为 `4.3/21`；相较 SJF，P99/Max service 降低 73.7%/88.8%，P95 max gap 降低 56.4%。不过它尚未赢得 makespan、吞吐或全局 completion tail。
- **结论是描述性的，不是统计或因果结论。** 每个条件目前只有一个 arrival seed/单次运行，且数据通过人工抄录获得；报告保留异常核验记录，不对小幅差异作过度解释。

## 2. 实验与口径

本报告比较 Ascend 910B4 五卡环境下，`0.03 req/s` 与 `0.12 req/s` 两档泊松到达率，以及 FCFS、SJF、RH-SAIL 三种调度策略。每个策略均完成 7 个 dataset × 50 条请求，共 `350/350`。

### 实验配置

| 项目 | 配置 |
|---|---|
| 硬件 | Ascend 910B4 × 5；两个 rate 分别运行在两台机器 |
| 模型 | Meta-Llama-3.1-8B-Instruct |
| 数据集 | `mixed_medium_first50.jsonl`，7 个 dataset × 50，共 350 请求 |
| 调度策略 | FCFS、SJF、RH-SAIL；同一机器上顺序运行 |
| 到达模式 | `poisson-burst`，batch size=1 |
| 到达率 | 机器 A：0.12 req/s；机器 B：0.03 req/s |
| Arrival seed | 20260709 |
| 输出配置 | medium，`output_max_tokens=2048` |
| 最大上下文 | `max_model_len=32768` |
| 显存比例 | `gpu_memory_utilization=0.75` |
| Prefix caching | 关闭，`MFE_ENABLE_PREFIX_CACHING=0` |
| 执行模式 | vLLM V1；`MFE_VLLM_ENFORCE_EAGER=1` |
| 生成采样 | 沿用模板默认值，temperature=0.7、top_p=0.9；未设置 per-request generation seed |

两档 rate 运行在不同物理机器上，三种策略只在同一 rate、同一机器内部共享硬件环境。后续 NPU 与 A800 报告只能把共同 rate `0.03/0.12` 作为直接硬件对照；A800 的 `0.15` 结果将作为额外高压参考，不与缺失 `0.15` 数据的 NPU 做同 rate 定量比较。

指标定义：

- `makespan`：首个请求到达到最后一个请求完成。
- `arrival end`：首个请求到达到最后一个请求到达。
- `drain`：最后一个请求到达后，系统排空剩余请求所需时间。
- `run time`：一个请求所有 op duration 之和，不包含排队等待和 op 间空档；并行 op 分别计时，因此可能大于 service time。
- `service`：请求首个 op 开始到请求完成，包含 DAG 内部等待和 op 间空档。
- `completion`：请求到达到请求完成，即等待时间加 service time。
- `P95 max gap`：先合并每个请求内重叠的 op 活跃区间，取该请求最大的相邻区间空档，再对全部请求取 P95。
- `ready`：调度器当前已暴露并可调度的 ready 节点数。RH-SAIL 存在 admission control，因此该指标表示 admitted-ready 压力，不等同于系统全部未完成工作量。

## 3. 数据质量、限制与稳健性检查

数据来自两台机器生成的 `detailed_brief.md` 的人工抄录，主体指标满足基本恒等关系。以下项目在分析前做了显式处理：

1. 用户复核原始 Markdown 后，将 `0.12/SJF` 更正为：Avg wait `2548.2s`、P99 service `2898.7s`、Max service `7331.2s`、Avg completion `2772.4s`，本报告采用更正值。此时 `Avg wait + Avg service = 2548.2 + 224.2 = 2772.4s`，指标完全闭合。
2. `0.03/math/FCFS` 的 service/run 已复核为 `183.1/172.1s`。
3. `0.03/mmlu_pro` 已复核为 FCFS `71.1/86.5s`、SJF `74.9/79.5s`、RH-SAIL `96.0/95.0s`。
4. `gms8k`、`mmlu_por`、`statrgyqa`、`swebench_verrified` 统一为正式 dataset 名；缺失的 `/` 仅做格式补全。

当前只有每个 rate × scheduler 一个运行结果，无法给出跨 seed 方差或置信区间。因此，本报告将数量级差异（例如 ready 降低 90% 以上、SJF Max service 达数千秒）视为稳健信号，将 1%--5% 的 makespan/吞吐差异视为需要重复实验确认的描述性现象。数据点只有 6 个系统聚合行，且读者需要核对精确尾部值，因此使用审计表格而非趋势图。

### 内部一致性检查通过

- 所有运行均为 `350/350`，且 `input tok/s + output tok/s = total tok/s`。
- 所有行均满足 `makespan ≈ arrival end + drain`，差异不超过一位小数的舍入误差。
- 所有策略均满足 `Avg completion ≈ Avg wait + Avg service`；更正后的 `0.12/SJF` 精确闭合。
- 所有尾部指标均满足 `P99 service <= Max service`、`P99 completion <= Max completion`。
- 两档 arrival end 的比值约为 4，符合相同泊松样本序列从 `0.03` 缩放到 `0.12 req/s` 的预期。

### 高风险：不同策略实际处理的 token 工作量不一致

根据一位小数的 tokens/s 与 makespan 反推，三策略的实际 token 总量存在无法由舍入解释的差异：

| Rate | 策略 | 估算 input tokens | 估算 output tokens | 估算 total tokens |
|---:|---|---:|---:|---:|
| 0.03 | FCFS | 6.930M | 0.787M | 7.717M |
| 0.03 | SJF | 6.729M | 0.756M | 7.484M |
| 0.03 | RH-SAIL | 6.815M | 0.789M | 7.604M |
| 0.12 | FCFS | 6.825M | 0.797M | 7.622M |
| 0.12 | SJF | 6.762M | 0.763M | 7.526M |
| 0.12 | RH-SAIL | 6.873M | 0.806M | 7.680M |

同一 rate 下，input tokens 的最大差异约为 1.6%--3.0%，output tokens 为 4.3%--5.7%。当前模型默认 `temperature=0.7、top_p=0.9`，且没有 per-request generation seed；生成内容和长度变化还会进入后续 DAG op 的 prompt，因而 input/output 工作量都会变化。尤其在 `0.12` 下，SJF 的 total token 工作量比 FCFS 少约 1.3%，而 makespan 仅短约 2.0%，因此其小幅 makespan 优势有相当部分可能来自工作量更少。正式对比应使用 greedy decoding 或稳定的 per-request seed，并直接引用 `detailed_brief_overall.csv` 中的精确 token 总数。

### 高风险：rate 与物理机器完全绑定

`0.03` 与 `0.12` 来自两台不同机器，因此“到达率效应”同时混入机器、驱动/CANN、vLLM-Ascend 版本、温度/频率和后台负载差异。排空时间相差一个数量级，过载判断不太可能只由机器差异造成；但 5%--10% 的 tokens/s 增益不能干净归因于 rate。最小修复是做交叉实验：机器 A/B 都跑一次 `0.03` 和 `0.12`，或至少让两台机器以相同 rate 跑一个短 FCFS calibration。

### 中风险：ready、调度开销和 device busy 不是直观口径

- RH-SAIL 的 ready 值受 admission control 限制，不能与 FCFS/SJF 的全部 ready backlog 等价解释。
- 调度开销在每个空闲 worker 的轮询中累计，包括 `ready` 为空的检查；低 rate 的到达窗口约长 4 倍，因此跨 rate 比较 `overhead seconds/%` 不等价于算法单次决策成本。应额外输出 decisions、non-empty decisions 和 overhead/decision。
- `device busy` 是 op duration 在 worker 上的占时比例，不是 `npu-smi` 的硬件利用率。硬件结论需要同时查看 NPU 采样数据。

## 4. 四倍到达率只带来个位数吞吐增益

| Rate | 策略 | 完成 | Makespan(s) | Arrival end(s) | Drain(s) | Input tok/s | Output tok/s | Total tok/s | Ready avg/peak | Device busy | 调度开销 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | FCFS | 350/350 | 12902.2 | 11929.5 | 972.8 | 537.1 | 61.0 | 598.1 | 1.3 / 23 | 89.0% | 215.6s / 1.67% |
| 0.03 | SJF | 350/350 | **12171.8** | 11929.5 | **242.3** | **552.8** | 62.1 | **614.9** | 1.3 / 28 | **90.4%** | **114.3s / 0.94%** |
| 0.03 | RH-SAIL | 350/350 | 12698.6 | 11929.4 | 769.2 | 536.7 | **62.1** | 598.8 | **0.6 / 21** | 90.0% | 219.0s / 1.72% |
| 0.12 | FCFS | 350/350 | 11805.4 | 2982.6 | 8822.8 | 578.1 | 67.5 | 645.6 | 121.5 / 257 | **96.5%** | **56.8s / 0.48%** |
| 0.12 | SJF | 350/350 | **11571.0** | 2982.5 | **8588.5** | **584.4** | 65.9 | **650.4** | 91.2 / 219 | 95.5% | 79.7s / 0.69% |
| 0.12 | RH-SAIL | 350/350 | 11928.4 | 2982.6 | 8945.7 | 576.2 | **67.6** | 643.8 | **4.3 / 21** | 96.1% | 167.2s / 1.40% |

### 到达率改变了什么

`0.03 req/s` 下，三个策略的 ready 均值不超过 1.3，排空需要 242--973 秒，同时 device busy 已达到 89%--90%。排空尾部相当于到达窗口的 2.0%（SJF）、6.4%（RH-SAIL）和 8.2%（FCFS），因此它更准确地属于“接近饱和、轻度积压但没有大规模 ready 堆积”，而不是完全空闲或严格不拥堵。SJF 最接近到达结束后同步排空。

`0.12 req/s` 将 350 个请求的到达窗口从约 3.31 小时压缩到约 49.7 分钟，但 makespan 仍约 3.2 小时，导致所有策略都需要额外 2.39--2.48 小时排空。相比 `0.03`，FCFS、SJF、RH-SAIL 的排空尾部分别放大到 `9.1x`、`35.4x`、`11.6x`，这是明确的持续过载状态。

四倍到达率只让 total tokens/s 提高约 5.8%--7.9%，output tokens/s 提高约 6.1%--10.7%，说明五张 910B4 已接近硬件吞吐上限。跨到达率直接比较 makespan 会产生误导：`0.12` 的 makespan略短，是因为请求更早全部到达，而不是系统更快；应主要看 drain、等待和完成时间。

## 5. SJF 赢平均值，FCFS 与 RH-SAIL 保护不同层面的尾部

| Rate | 策略 | Avg wait(s) | Avg run(s) | Avg service(s) | P99 service(s) | Max service(s) | Avg completion(s) | P99 completion(s) | Max completion(s) | P95 max gap(s) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | FCFS | 170.3 | 164.0 | **139.2** | **641.9** | 1097.2 | 309.5 | **969.2** | **1460.3** | **44.2** |
| 0.03 | SJF | **39.6** | **157.2** | 197.7 | 1636.0 | 4907.2 | **237.3** | 2505.0 | 5041.5 | 224.2 |
| 0.03 | RH-SAIL | 187.1 | 163.3 | 164.1 | 647.7 | **988.4** | 351.1 | 2214.3 | 3068.0 | 57.2 |
| 0.12 | FCFS | 4194.6 | 162.7 | **138.6** | **513.0** | 821.0 | 4333.2 | **8659.4** | **8854.0** | **53.8** |
| 0.12 | SJF | **2548.2** | **157.8** | 224.2 | 2898.7 | 7331.2 | **2772.4** | 9966.8 | 11042.2 | 132.5 |
| 0.12 | RH-SAIL | 2822.1 | 163.7 | 173.1 | 761.4 | **820.2** | 2995.2 | 9848.3 | 10341.6 | 57.8 |

三个策略的平均 run time 在同一负载下非常接近：`0.03` 为 157--164 秒，`0.12` 为 158--164 秒。这说明模型实际执行的 op 工作量基本相同，service 与 completion 的巨大差异主要来自调度顺序、DAG 内空档和首次等待，而不是某个策略让单个 op 计算显著变快。

### SJF：平均性能最好，但长请求尾部最弱

SJF 在两档负载下都观测到最短 makespan、最高 total tokens/s、最低平均等待和最低平均完成时间。它优先完成短工作，因此系统级平均指标最强；但由于 token 工作量不一致，这里只能确认观测结果，不能把全部差异解释为调度收益。

代价是长 DAG 容易被短工作持续插队。`0.03` 下 SJF 的 Max service 达 4907.2 秒，P95 max gap 达 224.2 秒；RH-SAIL 分别降低 `79.9%` 和 `74.5%`。`0.12` 下问题进一步放大，SJF 的 P99/Max service 达到 2898.7/7331.2 秒；RH-SAIL 分别降低 `73.7%/88.8%`，同时将 Max completion 降低 `6.3%`、P95 max gap 降低 `56.4%`，但平均完成时间增加 `8.0%`。

### FCFS：简单，但请求尾部异常稳健

FCFS 不重排已到达请求，天然避免长请求被后续短请求反复插队，因此两档负载下都取得最低 P99/Max completion 和最低 P95 max gap。`0.12` 下 FCFS 虽然平均完成时间比 RH-SAIL 高 `44.7%`，但 P99 completion 低 `12.1%`，Max completion 低 `14.4%`。

这解释了为什么 FCFS 在当前 workload 上很难被全面击败：请求长度差异大且每个 DAG 内含并行 op，简单到达序在牺牲平均等待的同时，提供了很强的全局 aging 与尾部公平性。

### RH-SAIL：优势是有界 ready 压力和连续性，不是最高吞吐

RH-SAIL 的最清晰优势出现在 `0.12` 过载区间：ready avg/peak 只有 `4.3/21`，相比 FCFS 的 `121.5/257` 分别降低 `96.5%/91.8%`，相比 SJF 的 `91.2/219` 分别降低 `95.3%/90.4%`。从 `0.03` 到 `0.12`，FCFS 与 SJF 的 ready 均值膨胀 `93.5x` 和 `70.2x`，RH-SAIL 仅增加到 `7.2x`。

但这表示 admission control 成功限制了当前暴露给调度器的 ready 集合，并不代表全部系统 backlog 消失。未被 admission 的请求仍会反映在等待、排空和完成时间中，因此 RH-SAIL 没有取得最短 makespan。它更适合被描述为“在过载下保持有界调度面，并减少已启动长 DAG 的中断”，而非“最大化吞吐”的策略。

RH-SAIL 的调度开销也最高：`0.12` 为 167.2 秒，占 makespan 1.40%，约为 FCFS 的 2.9 倍、SJF 的 2.1 倍。不过绝对占比仍低，当前主要差异仍来自调度选择与 admission，而非纯 CPU 调度开销。

## 6. RH-SAIL 对部分长 DAG 有效，但收益依赖 dataset

表格单元格均为 `平均 service / 平均 run time`，单位为秒。

### 0.03 req/s

| Dataset | Count | FCFS | SJF | RH-SAIL |
|---|---:|---:|---:|---:|
| gsm8k | 50 | 35.7 / 26.5 | **32.2 / 29.8** | 54.6 / 28.2 |
| hotpotqa | 50 | **119.5 / 118.6** | 170.1 / 110.6 | 140.9 / 114.7 |
| math | 50 | 183.1 / 172.1 | **148.2 / 142.2** | 214.1 / 165.9 |
| mbpp | 50 | 149.3 / 173.4 | **148.0 / 163.4** | 170.5 / 179.2 |
| mmlu_pro | 50 | **71.1 / 86.5** | 74.9 / 79.5 | 96.0 / 95.0 |
| strategyqa | 50 | 82.8 / 77.4 | **65.5 / 79.4** | 78.8 / 71.7 |
| swebench_verified | 50 | **333.2 / 493.2** | 745.0 / 495.3 | 393.7 / 488.2 |

### 0.12 req/s

| Dataset | Count | FCFS | SJF | RH-SAIL |
|---|---:|---:|---:|---:|
| gsm8k | 50 | **43.1 / 32.3** | 50.5 / 29.4 | 65.8 / 30.5 |
| hotpotqa | 50 | **112.2 / 112.5** | 394.7 / 111.2 | 156.8 / 124.0 |
| math | 50 | **161.1 / 145.9** | 196.1 / 153.9 | 211.5 / 159.0 |
| mbpp | 50 | **150.5 / 160.2** | 389.6 / 170.6 | 167.9 / 156.1 |
| mmlu_pro | 50 | **100.1 / 126.4** | 120.2 / 72.4 | 102.0 / 105.2 |
| strategyqa | 50 | **74.3 / 77.0** | 75.6 / 85.3 | 94.3 / 92.1 |
| swebench_verified | 50 | **328.7 / 484.8** | 342.7 / 481.7 | 413.5 / 479.1 |

Dataset 结果进一步说明 SJF 的长任务饥饿问题。`0.12` 下，RH-SAIL 相比 SJF 将 hotpotqa 平均 service 从 394.7 秒降至 156.8 秒，降低 `60.3%`；将 mbpp 从 389.6 秒降至 167.9 秒，降低 `56.9%`。`0.03` 下，RH-SAIL 将 swebench_verified 从 745.0 秒降至 393.7 秒，降低 `47.2%`。

不过 RH-SAIL 并非对所有 dataset 都更优。`0.12` 下 swebench_verified 的平均 service 比 SJF 高 `20.7%`，math 高 `7.9%`，且多数 dataset 仍不如 FCFS。当前策略的 continuity/admission 机制确实能救回一部分被 SJF 延后的长 DAG，但收益依赖 DAG 结构与到达时机。

## 7. 综合结论

1. **负载分档有效，但 `0.03` 不是完全空闲档。** 它是高利用率、低 ready 堆积、轻度排空尾部的近饱和档；`0.12 req/s` 是硬件吞吐饱和后的持续过载档。
2. **SJF 是观测平均性能和系统吞吐的赢家，但归因尚不干净。** 两档负载下，它都取得最短 makespan、最高 total tokens/s、最低平均等待和最低平均完成时间；实际 token 总量更少，使小幅 makespan 优势存在工作量混杂。
3. **FCFS 是尾部公平性的强基线。** 它在 P99/Max completion 与 P95 max gap 上持续领先，说明“无重排 + 自然 aging”非常适合当前长度差异大的 DAG workload。
4. **RH-SAIL 的贡献应聚焦在过载稳态与长请求连续性。** 它把 admitted-ready 集合保持在 21 个以内，并显著压低 SJF 的 Max service、Max completion 和最大 op 间空档；对 hotpotqa、mbpp 等长 DAG 尤其明显。
5. **RH-SAIL 仍存在明确改进空间。** 当前 admission 把 ready 压力转移成未 admission 等待，且 progress/rollout 没有同时赢过 FCFS 的全局尾部和 SJF 的平均性能。下一步应减少 rollout 开销，并引入显式的全局 arrival-age/未 admission backlog 代价，使“有界 ready”能够进一步转化为更短 drain 和 completion tail。

因此，基于这两档 Ascend 实验，更准确的结论不是“RH-SAIL 全面优于 FCFS/SJF”，而是：

> RH-SAIL 在过载下以很小的 admitted-ready 集合维持调度可控性，并显著缓解 SJF 对部分长 DAG 的 service starvation；其代价是更高调度复杂度，且尚未在系统吞吐、makespan 和全局 completion tail 上超过两个简单基线。

## 8. 建议与待验证问题

1. **保存原始 `detailed_brief.md/csv` 的屏幕记录。** 当前更正后的总体指标已满足均值恒等式；仍建议保留原始表格截图，便于后续审计人工补全的 dataset 行。
2. **记录 total backlog 与 admission backlog。** ready 指标只反映 admitted-ready 节点，下一轮应同时输出未 admission 请求数、全部 unfinished request 数及其 age，检验 RH-SAIL 是否真正减少系统积压，还是只缩小调度面。
3. **对 FCFS 的尾部优势做机制拆解。** 增加 request arrival rank、首次启动 rank、完成 rank 和 per-request DAG size，验证其优势来自自然 aging、设备亲和性，还是特定 dataset 顺序。
4. **至少增加 3 个 arrival seed。** 对 makespan、tokens/s、Avg/P99/Max completion 和 P95 max gap 报告均值与离散程度，再判断 1%--5% 的策略差异是否可复现。
5. **固定生成随机性。** 使用 greedy decoding 或 per-request 固定 seed，并记录各策略实际 input/output token 总数，避免生成长度差异混入调度比较。

进一步需要回答的核心问题是：RH-SAIL 能否在继续保持 ready 有界和 SJF 级长请求连续性的同时，引入全局 arrival-age/backlog 代价，逼近 SJF 的平均完成时间与 FCFS 的 completion tail。这个问题决定其优势能否从“调度面稳定”转化为端到端用户收益。

## 9. first50 A800 对照与后续报告

匹配配置的 NVIDIA A800 五卡实验现已完成，覆盖相同的 `mixed_medium_first50`、350 请求、arrival seed、FCFS/SJF/RH-SAIL，以及共同 rate `0.03/0.12`；另增加 `0.15` 作为 A800 饱和参考。九项 A800 实验均为 `350/350`、成功率 `100%`，且无 OOM、context、KV cache、Traceback、CUDA 或 NCCL 错误。

- [NVIDIA A800 first50 三速率调度性能报告](../20260717-a800-first50-rate-sweep/report.md)
- [Ascend 910B4 与 NVIDIA A800 first50 纯性能对比报告](../20260717-ascend-a800-first50-performance-comparison/report.md)

直接硬件比较只使用两侧匹配的 `0.03/0.12`。在 `0.12` 下，A800 的端到端 total tok/s 是 Ascend 的 `3.79--3.89×`，平均 op run time 快 `4.35--4.53×`，排空时间缩短 `79--135×`。在 `0.03` 下，端到端吞吐只提高 `3%--8%`，因为两侧 makespan 都由相同的长 arrival window 主导；此时 run time 仍显示 A800 快 `4.26--4.57×`。A800 `0.15` 没有匹配 NPU 数据，仅用于定位其饱和点，不做直接倍率比较。
