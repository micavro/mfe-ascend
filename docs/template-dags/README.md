# Template DAG diagrams

Generated from `templates/**/*.yaml`.

The combined bench overview uses the redesigned DAG set documented in `bench/dag_dataset_selection.md`.

| YAML | TikZ source | PDF |
| --- | --- | --- |
| `templates/bench/*.yaml overview` | `bench/bench_overview.tex` | `bench/bench_overview.pdf` |
| `templates/bench/*.yaml overview PNG` | `bench/bench_overview.png` | `bench/bench_overview.png` |
| `templates/bench/agentic_repair_swebench_verified_medium.yaml` | `bench/agentic_repair_swebench_verified_medium.tex` | `bench/agentic_repair_swebench_verified_medium.pdf` |
| `templates/bench/branch_verify_strategyqa_medium.yaml` | `bench/branch_verify_strategyqa_medium.tex` | `bench/branch_verify_strategyqa_medium.pdf` |
| `templates/bench/chain_gsm8k_medium.yaml` | `bench/chain_gsm8k_medium.tex` | `bench/chain_gsm8k_medium.pdf` |
| `templates/bench/debate_mmlu_pro_medium.yaml` | `bench/debate_mmlu_pro_medium.tex` | `bench/debate_mmlu_pro_medium.pdf` |
| `templates/bench/plan_code_test_mbpp_medium.yaml` | `bench/plan_code_test_mbpp_medium.tex` | `bench/plan_code_test_mbpp_medium.pdf` |
| `templates/bench/research_panel_gpqa_diamond_medium.yaml` | `bench/research_panel_gpqa_diamond_medium.tex` | `bench/research_panel_gpqa_diamond_medium.pdf` |
| `templates/bench/self_refine_math_medium.yaml` | `bench/self_refine_math_medium.tex` | `bench/self_refine_math_medium.pdf` |

Regenerate diagrams with:

```bash
python -m mfe.scripts.generate_template_dags --bench-only --bench-overview --compile-pdf --clean
```
