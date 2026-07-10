# Helium: Efficient LLM Serving for Agentic Workflows

## 基本信息

- 论文：Efficient LLM Serving for Agentic Workflows: A Data Systems Perspective
- arXiv：2603.16104v1
- 作者：Noppanat Wadlom, Junyi Shen, Yao Lu
- 本地 PDF：`docs/reports/papers/helium-agentic-workflows-2603.16104v1.pdf`
- 本地 Figure 6 截图：`docs/reports/papers/helium-trading-workflow-figure6.png`
- 主题：把 agentic workflow 建模成可优化的查询计划 DAG，在工作流层面做 LLM serving、缓存和调度。

## 一句话总结

这篇论文的核心不是提出新的推理算法，而是提出一个系统视角：把一个 agent workflow 里的每次 LLM 调用当作 query plan 里的 operator，用 DAG 显式表达依赖关系，然后利用 DAG 结构做 plan rewriting、prompt/KV cache 复用和 cache-aware scheduling。

## 论文要解决的问题

传统 LLM serving 系统一般把请求当作独立调用来优化，重点在单次请求的 batching、prefill/decode、KV 管理等。agentic workflow 不一样：一个任务通常包含多个 LLM agent、多轮调用、并行分支、汇总节点、反思/辩论节点，以及多个请求之间可共享的静态上下文或 prompt prefix。

论文认为现有系统主要缺三个层面的全局视角：

1. Operator abstraction：LLM 调用没有被当成 workflow operator 来统一建模，导致系统不知道哪些调用彼此依赖、哪些可以并行。
2. Inter-operator sharing：同一个 workflow 内部的多个 operator 可能共享 prompt 前缀、上下文、系统提示或静态输入，但普通 serving 只能局部复用。
3. Inter-workflow sharing：不同 query 或不同 batch 之间也可能重复使用相同资料，例如公司基本面、固定文档、模板化 prompt，这些可以提前缓存或转成 CacheFetch。

## Helium 的系统架构

Helium 把 agent workflow 编译成一个符号 DAG。用户用 Python DSL 写 workflow，但 operator 调用不会立刻执行，而是生成节点和依赖边。之后系统执行三步：

1. 构建初始 DAG：每个 LLM 调用、输入占位符、汇总/中间结果都是显式节点。
2. Query optimizer 优化逻辑计划：删除无用节点、合并公共子图、用缓存命中的 `CacheFetch` 替换重复计算。
3. Query processor 执行物理计划：构建 Templated Radix Tree 表示 prompt prefix 结构，结合 proactive KV/prompt cache 和 cache-aware scheduler 分配执行顺序与 worker。

系统实现上基于 vLLM，增强点包括 KV cache pinning、预填充固定前缀、缓存命中后的 plan rewrite，以及根据 operator 依赖和 prefix 复用机会进行调度。

## Figure 2 的五类基础 DAG

论文用五类 primitive agentic workflow 做 microbenchmark。师兄说“组合论文时使用这个论文的 DAG 架构来做”，最应该对齐的就是这五类基础形态。

### W1 Map-Reduce

结构：多个 expert agent 并行处理同一个问题或不同上下文，最后由 summarizer 汇总。

特点：

- fan-out / fan-in 明显；
- 并行度高；
- 汇总节点依赖所有专家输出；
- 很适合测并行调度、prefix 复用和聚合阶段等待时间。

论文中使用 MMLU、TAT-QA 一类 QA 数据构造。

### W2 Multi-Agent Debate

结构：多个 agent 围绕同一问题并行或多轮给出立场、反驳、修正，最后由 judge 或 moderator 汇总答案。

特点：

- 多个 debater 共享题目和上下文；
- 多轮结构会重复使用大量 prompt prefix；
- 适合测多分支、多轮依赖、共享 prefix 缓存和最终裁决。

论文中使用 MMLU、TAT-QA 一类问题。对我们来说，MMLU-Pro 或 GPQA 这种高难选择题更适合做 debate，因为多个候选推理路径确实有意义。

### W3 Multi-Agent Reflection

结构：一个 expert 先给 draft answer，多个 critic 再审查，最后 refinement agent 根据反馈修正。

特点：

- 不是简单并行投票，而是 draft -> critique -> revise；
- critic 节点可以并行，revise 节点依赖 critic 输出；
- 适合测分支反馈、反思修正和中间结果复用。

它和 Debate 的区别在语义上很重要：Debate 是多个候选答案/立场竞争，Reflection 是单个答案被审查和修正。

### W4 Iterative Refinement

结构：输入被切成多个 chunk，一个 summarizer 或 refinement agent 按顺序处理，每一步基于上一步的 summary 更新结果。

特点：

- 长链式依赖；
- 并行度低，但状态传递强；
- 适合测长依赖链、逐步压缩、跨 chunk 上下文累积。

论文中用 Amazon Reviews 这类长文本评论数据构造。这个形态和普通 self-refine 不完全相同：它强调“分块输入上的顺序累积”，不是单个答案的 critic/revise。

### W5 Parallel Chains

结构：多个专家链并行运行，每条链负责不同角色或不同输入块，最后 writer 汇总成报告。

特点：

- 每个分支内部可以是 chain；
- 分支之间并行；
- 末端有全局汇总；
- 比 Map-Reduce 多了“分支内多步链式处理”。

论文中也用 Amazon Reviews 构造，例如多个角色从不同评论块抽取 insight，再由 writer 写成报告。

## Figure 6 的复合 DAG：Trading workflow

论文的复杂端到端 benchmark 是 Trading workflow。它不是随便把几个 DAG 拼起来，而是用一个真实应用语义把多个 primitive 组合起来：

1. Analyst stage：Parallel pattern。market、social media、news、fundamentals 等 analyst 分别处理不同金融数据源。
2. Research stage：Debate pattern。bullish / bearish researcher 围绕 analyst 输出辩论，由 manager 组织。
3. Decision stage：Map-Reduce pattern with nested complexity。多个 trader 分支给出策略，再经过 risk management debate，最后 fund manager 汇总成投资建议。

这个 workflow 包含 19 个 agents、88 个 LLM operators。它的关键价值是：不同阶段同时包含并行分支、辩论、多轮风险评估、最终汇总、跨 batch 静态数据复用。也就是说，复杂 DAG 的复杂性来自任务本身，而不是为了混合而混合。

## 主要优化机制

### Logical plan optimization

- Operator pruning：删除最终输出不需要的 operator。
- Common subgraph elimination：合并重复子图，避免相同计算重复执行。
- Prompt cache substitution：如果某个子图或 operator 输出已经在 prompt cache 中，直接用 `CacheFetch` 节点替换。

### Templated Radix Tree

Helium 用 Templated Radix Tree 表示 prompt 模板、静态前缀和动态 placeholder 的结构。相比普通按字符串缓存，它能更清楚地区分哪些部分是多个请求共享的模板，哪些部分是 query-specific 输入。

### Proactive caching

系统会提前预计算长静态前缀并保留 KV cache，或者缓存完整 prompt/operator 输出。这样后续 batch 到来时，可以少做 prefill 或直接跳过某些 operator。

### Cache-aware scheduling

调度器不是只看 ready queue 或最短任务，而是同时看：

- DAG 依赖；
- worker 负载；
- operator 执行成本；
- prefix/KV cache 命中机会；
- 当前调度是否会增加或破坏后续缓存复用。

直观上，它会在“尽快并行执行”和“把共享前缀放在一起执行”之间做权衡。

## 实验结论

论文报告的核心结果：

- 在五类 primitive workflow 上，Helium 相比 KVFlow 最高达到 1.56x 加速。
- 在 Trading 复合 workflow 上，Helium 相比 naive vLLM baseline 最高减少 39.50x latency。
- 在 Trading 上，相比 KVFlow 最高达到 1.34x 加速。
- Ablation 显示各组件都有贡献：去掉 plan pruning 性能下降 23.35%，去掉 cache-aware scheduling 下降 17.66%，去掉 prompt caching 下降 13.56%，去掉 proactive KV caching 下降 3.55%。
- Scheduling-only 对比中，Helium 的最优性差距平均 0.9%，最大 3.6%，明显优于 QueryWise、OpWise、Random、LSPF 等基线。

## 对我们项目的直接启发

这篇论文给了我们一个更稳的 benchmark 组织方式。我们之前设计 DAG 时已经有 chain、diamond、debate、self-refine、code-test、research panel、repair 等形态，但如果要写进论文或和师兄讨论，建议改成“先对齐 Helium primitive taxonomy，再扩展自己的系统贡献”。

推荐后续组织方式：

1. 基础 DAG 不再按我们自己随意命名的任务分类，而是对齐 Helium Figure 2：
   - Map-Reduce
   - Multi-Agent Debate
   - Multi-Agent Reflection
   - Iterative Refinement
   - Parallel Chains
2. 复杂 DAG 至少做一个类似 Figure 6 的真实应用复合 workflow：
   - Parallel -> Debate -> Map-Reduce；
   - 复杂性来自真实任务阶段，而不是人为把所有结构拼一起。
3. 数据集选择应该服从 DAG 语义：
   - Debate：高难多选或科学问题，如 MMLU-Pro / GPQA；
   - Reflection：需要先答后批改的推理或数学题，如 MATH / TAT-QA；
   - Iterative Refinement：长文档或多 chunk 汇总，如 Amazon Reviews / long-document summarization；
   - Parallel Chains：多源信息抽取后写报告，如评论分析、金融研报、多文档 QA；
   - Map-Reduce：多专家并行解题或多文档并行分析后汇总。
4. 我们的论文卖点如果是调度/serving，需要明确比较对象：
   - 普通 vLLM：只做请求级 serving；
   - OpWise / QueryWise 类策略：知道 DAG 但不会全局优化缓存；
   - Helium 类思路：workflow-level DAG + cache-aware scheduling；
   - 我们自己的贡献：需要说明是在 Ascend/CUDA 统一部署、多卡调度、实验任务生成、还是某种新的 DAG scheduling 策略上超过或补充 Helium。

## 需要注意的边界

Helium 也指出 DAG abstraction 对动态控制流有限制，例如 conditional loop、dynamic mapping、外部 API 调用延迟等。这对我们有用：如果我们当前 benchmark 主要是静态 DAG，就可以明确说第一阶段聚焦 static agentic DAG；如果要扩展到动态 agent workflow，需要单独设计控制流和运行时 profiling。

## 和当前 7 个 benchmark 的关系

当前 7 个 benchmark 可以保留为任务候选，但建议重排到 Helium taxonomy 下：

- `debate_mmlu_pro_medium.yaml` 可以直接对应 Multi-Agent Debate。
- `self_refine_math_medium.yaml` 更接近 Multi-Agent Reflection，而不是 Iterative Refinement。
- `chain_gsm8k_medium.yaml` 是普通 chain，Helium Figure 2 里没有把它作为核心 primitive；可以作为 sanity baseline，而不是主打 DAG 类型。
- `branch_verify_strategyqa_medium.yaml` 接近 diamond / verification，但不属于 Helium 的五类核心 primitive；若保留，需要解释它是我们额外加入的 verification DAG。
- `plan_code_test_mbpp_medium.yaml` 是软件工程 agent chain，可作为应用型 workflow，但不应替代 Helium 的 Iterative Refinement。
- `research_panel_gpqa_diamond_medium.yaml` 可以改造成 Parallel + Debate + Reflection 的复合科学问答 workflow。
- `agentic_repair_swebench_verified_medium.yaml` 可以作为更难的真实软件修复复合 workflow，但它和 Helium Trading 的形态不同，需要单独解释数据准备和执行代价。

我的建议是：下一版图和 YAML 可以以 Helium 五类 primitive 为骨架，再额外保留 1 个 Trading-style 大复合 DAG 和 1 个软件修复/科研问答复合 DAG。这样更容易解释“为什么这些 DAG 是经典形态”，也更容易和已有工作对齐。
