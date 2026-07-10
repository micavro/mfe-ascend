# Design-7 benchmark workloads

This directory contains fixed medium-output JSONL workloads for the redesigned 7-DAG benchmark set.
The mixed workload order is a deterministic global random shuffle across all
selected rows, so arrivals are not grouped by dataset A, then dataset B, etc.

Generation command:

```bash
python -m mfe.scripts.build_experiment_datasets --data-dir data --output-dir data/experiments_design7 --size first50 --output-length medium --seed 20260709
python -m mfe.scripts.build_experiment_datasets --data-dir data --output-dir data/experiments_design7 --size first100 --output-length medium --seed 20260709
python -m mfe.scripts.build_experiment_datasets --data-dir data --output-dir data/experiments_design7 --size first200 --output-length medium --seed 20260709
```

Dataset to DAG mapping:

| Dataset | YAML |
| --- | --- |
| `gsm8k` | `bench/chain_gsm8k_medium.yaml` |
| `strategyqa` | `bench/branch_verify_strategyqa_medium.yaml` |
| `mmlu_pro` | `bench/debate_mmlu_pro_medium.yaml` |
| `math` | `bench/self_refine_math_medium.yaml` |
| `mbpp` | `bench/plan_code_test_mbpp_medium.yaml` |
| `hotpotqa` | `bench/parallel_debate_mapreduce_hotpotqa_medium.yaml` |
| `swebench_verified` | `bench/agentic_repair_swebench_verified_medium.yaml` |

Counts:

| Size | Per-dataset files | Mixed rows |
| --- | --- | --- |
| `first50` | 7 datasets x 50 | 350 |
| `first100` | 7 datasets x 100 | 700 |
| `first200` | 7 datasets x 200 | 1400 |

The companion package is `data/mfe_design7_medium_50_100_200.zip`.
