# Bench DAG and dataset selection

这版 bench 先固定 DAG 形态，再为每个形态选择匹配的数据集。旧的 `path_4_medium.yaml`、`large_mixed_medium.yaml` 等模板仍保留，避免破坏已有脚本；总览图只展示下面 7 个新设计模板。

| 类型 | YAML | DAG 特性 | 推荐数据集 | 选择理由 |
| --- | --- | --- | --- | --- |
| 基础 | `chain_gsm8k_medium.yaml` | 线性链式推理 | GSM8K | 小学数学文字题，天然适合 parse -> solve -> check -> final。 |
| 基础 | `branch_verify_strategyqa_medium.yaml` | 隐式分解 + 双分支校验 | StrategyQA | yes/no 常识问题常需要发现隐藏推理链；support/challenge 两路能检查假设。 |
| 基础 | `debate_mmlu_pro_medium.yaml` | 多路并行辩论 + 裁决 | MMLU-Pro | 多任务高难选择题，选项更多、干扰项更强，适合多个 debater 独立作答后由 judge 裁决。 |
| 基础 | `self_refine_math_medium.yaml` | 草稿 - 批改 - 修正 | MATH | 竞赛数学推导容易出错，适合 critic/revise/verify 结构。 |
| 基础 | `plan_code_test_mbpp_medium.yaml` | 计划 - 代码 - 测试 | MBPP | 小型 Python 编程任务带自然语言需求和测试，适合 plan/code/test-review。 |
| 复杂 | `research_panel_gpqa_diamond_medium.yaml` | 专家组 + 证据 + 辩论 + 反思 | GPQA Diamond | 研究生级科学选择题，适合多专家、证据合并、选项排除、辩论和反思。 |
| 复杂 | `agentic_repair_swebench_verified_medium.yaml` | 定位 + 双补丁 + 测试选择 | SWE-bench Verified | 真实 GitHub issue 修复，适合定位、补丁候选、验证、选择和回归风险检查。 |

当前 workload builder 已支持这 7 个数据集到 DAG 的映射。已生成的固定采样 workload 在 `data/experiments_design7/`，打包文件为 `data/mfe_design7_medium_50_100_200.zip`。GPQA Diamond 公开镜像只有 198 条，因此 50/100 档包含它，200 档按“不够就不塞”的规则跳过。

参考来源：

- GSM8K: https://arxiv.org/abs/2110.14168
- StrategyQA: https://arxiv.org/abs/2101.02235
- MMLU-Pro: https://arxiv.org/abs/2406.01574
- MATH: https://arxiv.org/abs/2103.03874
- MBPP: https://arxiv.org/abs/2108.07732
- GPQA: https://arxiv.org/abs/2311.12022
- SWE-bench: https://arxiv.org/abs/2310.06770
