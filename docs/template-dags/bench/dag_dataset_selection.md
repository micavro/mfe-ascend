# Bench DAG and dataset selection

这版 bench 先固定 DAG 形态，再为每个形态选择匹配的数据集。旧的 `path_4_medium.yaml`、`large_mixed_medium.yaml` 等模板仍保留，避免破坏已有脚本；总览图只展示下面 7 个新设计模板。

| 类型 | YAML | DAG 特性 | 推荐数据集 | 选择理由 |
| --- | --- | --- | --- | --- |
| 基础 | `chain_gsm8k_medium.yaml` | 线性链式推理 | GSM8K | 小学数学文字题，天然适合 parse -> solve -> check -> final。 |
| 基础 | `branch_verify_strategyqa_medium.yaml` | 隐式分解 + 双分支校验 | StrategyQA | yes/no 常识问题常需要发现隐藏推理链；support/challenge 两路能检查假设。 |
| 基础 | `debate_mmlu_pro_medium.yaml` | 多路并行辩论 + 裁决 | MMLU-Pro | 多任务高难选择题，选项更多、干扰项更强，适合多个 debater 独立作答后由 judge 裁决。 |
| 基础 | `self_refine_math_medium.yaml` | 两轮数学自我批改 | MATH | 竞赛数学推导容易出错；start -> draft -> critic -> revise -> critic -> revise -> critic -> final 更接近人工反复验算。 |
| 基础 | `plan_code_test_mbpp_medium.yaml` | 计划 - 代码 - 测试 | MBPP | 小型 Python 编程任务带自然语言需求和测试，适合 plan/code/test-review。 |
| 复杂 | `parallel_debate_mapreduce_hotpotqa_medium.yaml` | 四路并行 + 辩论 + Map-Reduce | HotpotQA | 多文档多跳问答；四路证据分析先找桥接、比较、直接答案和干扰项，再辩论候选答案，最后把答案、证据、推理链和一致性检查 Map-Reduce 汇总。 |
| 复杂 | `agentic_repair_swebench_verified_medium.yaml` | 定位 + 双补丁 + 测试选择 | SWE-bench Verified | 真实 GitHub issue 修复，适合定位、补丁候选、验证、选择和回归风险检查。 |

当前 workload builder 已支持这 7 个数据集到 DAG 的映射。已生成的固定采样 workload 在 `data/experiments_design7/`，打包文件为 `data/mfe_design7_medium_50_100_200.zip`。HotpotQA 本地镜像数量足够，因此 50/100/200 三档都包含这个复合 DAG。

Prompt 检查记录：GSM8K、StrategyQA、MMLU-Pro、MBPP、SWE-bench Verified 的既有 prompt 与各自任务目标匹配，本次不改拓扑；MATH 按两轮 refine 目标扩成显式多轮批改/修正；HotpotQA 新模板按先设计 DAG、再选多文档多跳数据集、再填证据分析/辩论/Map-Reduce prompt 的顺序构造。

参考来源：

- GSM8K: https://arxiv.org/abs/2110.14168
- StrategyQA: https://arxiv.org/abs/2101.02235
- MMLU-Pro: https://arxiv.org/abs/2406.01574
- MATH: https://arxiv.org/abs/2103.03874
- MBPP: https://arxiv.org/abs/2108.07732
- HotpotQA: https://arxiv.org/abs/1809.09600
- SWE-bench: https://arxiv.org/abs/2310.06770
