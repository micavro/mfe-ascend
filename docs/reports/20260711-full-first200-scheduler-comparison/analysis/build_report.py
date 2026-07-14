#!/usr/bin/env python3
"""Build the UTF-8 Markdown report from generated CSV/JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


RUN_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = RUN_ROOT / "report.md"
SCHEDULERS = ["fcfs", "sjf", "sailp"]
LABELS = {"fcfs": "FCFS", "sjf": "SJF", "sailp": "SAILP"}
DATASET_ORDER = [
    "gsm8k",
    "strategyqa",
    "mmlu_pro",
    "math",
    "mbpp",
    "hotpotqa",
    "swebench_verified",
]


def markdown_table(headers: list[str], rows: Iterable[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def summary_json(scheduler: str) -> dict:
    matches = list((RUN_ROOT / scheduler).glob("*_run1_summary.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one summary for {scheduler}, found {matches}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value:+.2f}%"


def build_report(run_root: Path = RUN_ROOT) -> Path:
    global RUN_ROOT, REPORT_PATH
    RUN_ROOT = run_root
    REPORT_PATH = RUN_ROOT / "report.md"

    metrics = pd.read_csv(RUN_ROOT / "analysis/scheduler_metrics.csv", index_col=0)
    device = pd.read_csv(RUN_ROOT / "analysis/device_busy_comparison.csv")
    dataset = pd.read_csv(RUN_ROOT / "analysis/dataset_metrics.csv")
    relative = pd.read_csv(RUN_ROOT / "analysis/relative_vs_fcfs.csv")
    manifest = json.loads((RUN_ROOT / "manifest_medium_first200.json").read_text(encoding="utf-8"))
    summaries = {scheduler: summary_json(scheduler) for scheduler in SCHEDULERS}

    if set(metrics.index) != set(SCHEDULERS):
        raise RuntimeError(f"Unexpected schedulers: {metrics.index.tolist()}")
    if not ((metrics["completed"] == 1400).all() and (metrics["success_rate"] == 1.0).all()):
        raise RuntimeError("One or more runs are incomplete")

    first = summaries["fcfs"]
    run_info = first["run_info"]

    core_headers = [
        "\u7b56\u7565",
        "\u5b8c\u6210",
        "\u7cfb\u7edf\u603b\u5b8c\u6210\u65f6\u95f4",
        "\u603b tokens",
        "\u8f93\u51fa tokens",
        "\u8bf7\u6c42\u541e\u5410\u91cf",
        "\u603b\u541e\u5410\u91cf",
        "\u8f93\u51fa\u541e\u5410\u91cf",
    ]
    core_rows = []
    for scheduler in SCHEDULERS:
        row = metrics.loc[scheduler]
        core_rows.append(
            [
                LABELS[scheduler],
                f"{int(row['completed'])}/{int(row['count'])}",
                f"{row['makespan']:.1f}s ({row['makespan_minutes']:.1f}min)",
                f"{int(row['total_tokens']):,}",
                f"{int(row['output_tokens']):,}",
                f"{row['request_throughput']:.4f} req/s",
                f"{row['total_token_throughput']:.1f} token/s",
                f"{row['output_token_throughput']:.1f} token/s",
            ]
        )

    timing_headers = [
        "\u7b56\u7565",
        "\u5e73\u5747\u7b49\u5f85",
        "\u5e73\u5747 service time",
        "P50 service time",
        "P95 service time",
        "P99 service time",
        "\u5e73\u5747\u5b8c\u6210\u65f6\u95f4",
    ]
    timing_rows = []
    for scheduler in SCHEDULERS:
        row = metrics.loc[scheduler]
        timing_rows.append(
            [
                LABELS[scheduler],
                f"{row['waiting_time_mean']:.1f}s",
                f"{row['service_time_mean']:.1f}s",
                f"{row['p50_service_time_s']:.1f}s",
                f"{row['p95_service_time_s']:.1f}s",
                f"{row['p99_service_time_s']:.1f}s",
                f"{row['avg_completion_time_s']:.1f}s",
            ]
        )

    scheduling_headers = [
        "\u7b56\u7565",
        "ready queue avg / peak",
        "dependency stall",
        "\u8c03\u5ea6\u5f00\u9500",
        "critical path",
        "DAG parallelism",
        "\u8de8 device \u4f9d\u8d56",
        "\u8d1f\u8f7d\u4e0d\u5747\u8861",
        "\u5e76\u884c\u5229\u7528\u7387",
    ]
    scheduling_rows = []
    for scheduler in SCHEDULERS:
        row = metrics.loc[scheduler]
        scheduling_rows.append(
            [
                LABELS[scheduler],
                f"{row['ready_queue_avg']:.1f} / {int(row['ready_queue_peak'])}",
                f"{row['dependency_stall_mean']:.1f}s",
                f"{row['scheduler_overhead_seconds']:.1f}s ({row['scheduler_overhead_pct'] * 100:.2f}%)",
                f"{row['critical_path_mean']:.1f}s",
                f"{row['dag_parallelism_mean']:.3f}",
                f"{row['cross_device_dependencies_mean']:.2f}",
                f"{row['load_imbalance']:.4f}",
                f"{row['parallelism_utilization'] * 100:.2f}%",
            ]
        )

    relative_names = {
        "system_makespan": "\u7cfb\u7edf\u603b\u5b8c\u6210\u65f6\u95f4",
        "total_token_throughput": "\u603b token \u541e\u5410\u91cf",
        "output_token_throughput": "\u8f93\u51fa token \u541e\u5410\u91cf",
        "average_waiting_time": "\u5e73\u5747\u7b49\u5f85\u65f6\u95f4",
        "average_service_time": "\u5e73\u5747 service time",
        "average_completion_time": "\u5e73\u5747\u5b8c\u6210\u65f6\u95f4",
        "ready_queue_peak": "ready queue peak",
        "scheduler_overhead": "\u8c03\u5ea6\u5f00\u9500",
        "dependency_stall": "dependency stall",
    }
    relative_headers = ["\u6307\u6807", "\u4f18\u9009\u65b9\u5411", "SJF vs FCFS", "SAILP vs FCFS"]
    relative_rows = []
    for row in relative.itertuples(index=False):
        relative_rows.append(
            [
                relative_names[str(row.metric)],
                "\u8d8a\u4f4e\u8d8a\u597d" if row.preferred_direction == "lower" else "\u8d8a\u9ad8\u8d8a\u597d",
                pct(float(row.sjf_vs_fcfs_pct)),
                pct(float(row.sailp_vs_fcfs_pct)),
            ]
        )

    device_headers = ["Physical GPU", "FCFS", "SJF", "SAILP"]
    device_rows = []
    for gpu in range(3, 8):
        values = []
        for scheduler in SCHEDULERS:
            value = device[
                (device["scheduler"] == scheduler) & (device["physical_gpu"] == gpu)
            ]["running_time_pct"].iloc[0]
            values.append(f"{value:.2f}%")
        device_rows.append([str(gpu), *values])

    dataset_headers = ["Dataset", "FCFS", "SJF", "SAILP"]
    dataset_rows = []
    for name in DATASET_ORDER:
        values = []
        for scheduler in SCHEDULERS:
            value = dataset[
                (dataset["scheduler"] == scheduler) & (dataset["dataset"] == name)
            ]["avg_service_time_s"].iloc[0]
            values.append(f"{value:.1f}s")
        dataset_rows.append([name, *values])

    fcfs = metrics.loc["fcfs"]
    sjf = metrics.loc["sjf"]
    sailp = metrics.loc["sailp"]
    token_spread = (
        (metrics["total_tokens"].max() - metrics["total_tokens"].min())
        / metrics["total_tokens"].mean()
        * 100
    )

    report = f"""# Full first200 \u8c03\u5ea6\u7b56\u7565\u7cfb\u7edf\u6027\u80fd\u62a5\u544a

## \u5b9e\u9a8c\u914d\u7f6e

| \u9879\u76ee | \u914d\u7f6e |
| --- | --- |
| \u6570\u636e\u89c4\u6a21 | 7 \u4e2a dataset\uff0c\u6bcf\u4e2a\u968f\u673a\u9009\u53d6 200 \u6761\uff0c\u5171 1,400 requests |
| \u6570\u636e\u987a\u5e8f | \u957f\u5ea6\u8fc7\u6ee4\u540e\u6309 seed {manifest['seed']} \u968f\u673a\u62bd\u6837\uff0cmixed workload \u5168\u5c40\u968f\u673a\u6253\u4e71 |
| \u6570\u636e\u96c6 | {', '.join(manifest['datasets'])} |
| \u957f\u5ea6\u8fc7\u6ee4 | start prompt < 14000 tokens\uff1bstart prompt + op max tokens <= 15500 |
| \u6a21\u578b | {run_info['model_path']} |
| \u63a8\u7406\u540e\u7aef | real vLLM {run_info['packages']['vllm']}\uff0cCUDA / NVIDIA A800 |
| GPU | physical GPU 3,4,5,6,7\uff0c\u5171 5 \u5f20 |
| max model len | {run_info['max_model_len']} |
| output max tokens | {first['output_max_tokens']} per DAG op |
| GPU memory utilization | {run_info['gpu_memory_utilization']} |
| \u5230\u8fbe\u8fc7\u7a0b | Poisson 0.15 req/s\uff0carrival batch size 1\uff0cseed {first['arrival_seed']} |
| \u8c03\u5ea6\u7b56\u7565 | FCFS\u3001SJF\u3001SAILP |
| \u4ee3\u7801\u7248\u672c | {run_info['git_commit']} |

## \u6838\u5fc3\u7cfb\u7edf\u6307\u6807

{markdown_table(core_headers, core_rows)}

![Scheduler metrics](figures/scheduler_metrics_comparison.png)

## \u8bf7\u6c42\u65f6\u95f4\u6307\u6807

{markdown_table(timing_headers, timing_rows)}

\u5e73\u5747\u7b49\u5f85\u8868\u793a request \u5230\u8fbe\u540e\u5230\u7b2c\u4e00\u4e2a op \u5f00\u59cb\uff1bservice time \u8868\u793a\u7b2c\u4e00\u4e2a op \u5f00\u59cb\u5230\u6574\u4e2a DAG \u5b8c\u6210\u3002\u4e8c\u8005\u5fc5\u987b\u540c\u65f6\u89c2\u5bdf\u3002

## \u8c03\u5ea6\u4e0e\u961f\u5217\u6307\u6807

{markdown_table(scheduling_headers, scheduling_rows)}

## \u76f8\u5bf9 FCFS \u7684\u53d8\u5316

{markdown_table(relative_headers, relative_rows)}

\u767e\u5206\u6bd4\u8ba1\u7b97\u65b9\u5f0f\u4e3a\uff1a\u7b56\u7565\u503c / FCFS - 1\u3002\u65f6\u95f4\u3001\u961f\u5217\u548c stall \u6307\u6807\u901a\u5e38\u8d8a\u4f4e\u8d8a\u597d\uff1b\u541e\u5410\u91cf\u8d8a\u9ad8\u8d8a\u597d\u3002

## Device \u5360\u7528\u65f6\u95f4\u7ebf

\u7eff\u8272\u8868\u793a\u8be5 device \u6b63\u5728\u6267\u884c\u81f3\u5c11\u4e00\u4e2a DAG op\uff1b\u7a7a\u767d\u8868\u793a\u6ca1\u6709 op \u8fd0\u884c\u3002\u5185\u90e8 device 0-4 \u6620\u5c04\u5230 physical GPU 3-7\u3002\u865a\u7ebf\u662f\u6309 1400 / 0.15 \u8ba1\u7b97\u7684\u7406\u8bba\u6cca\u677e\u5230\u8fbe\u7a97\u53e3\u672b\u7aef\u3002

![Device occupancy comparison](figures/device_occupancy_comparison.png)

{markdown_table(device_headers, device_rows)}

![Device busy comparison](figures/device_busy_comparison.png)

## Dataset \u5e73\u5747 Service Time

{markdown_table(dataset_headers, dataset_rows)}

![Dataset service time comparison](figures/dataset_service_time_comparison.png)

## \u7ed3\u8bba

- \u4e09\u79cd\u7b56\u7565\u5747\u5b8c\u6210 1400/1400\uff0c\u6210\u529f\u7387 100%\uff0c\u6ca1\u6709 context length\u3001OOM \u6216\u8fd0\u884c\u5d29\u6e83\u3002
- SJF \u7684\u7cfb\u7edf\u603b\u5b8c\u6210\u65f6\u95f4\u6700\u77ed\uff1a{sjf['makespan']:.1f}s\uff0c\u6bd4 FCFS \u7f29\u77ed {abs(float(relative.loc[relative['metric'] == 'system_makespan', 'sjf_vs_fcfs_pct'].iloc[0])):.2f}%\uff1b\u603b\u541e\u5410\u91cf\u4e3a {sjf['total_token_throughput']:.1f} token/s\u3002
- FCFS \u7684\u5e73\u5747\u7b49\u5f85\u8f83\u9ad8\uff1a{fcfs['waiting_time_mean']:.1f}s\uff0c\u4f46\u5e73\u5747 service time \u6700\u77ed\uff1a{fcfs['service_time_mean']:.1f}s\u3002
- SJF \u7684\u5e73\u5747\u5b8c\u6210\u65f6\u95f4\u4e3a {sjf['avg_completion_time_s']:.1f}s\uff0c\u660e\u663e\u4f4e\u4e8e FCFS\uff1b\u4f46 P99 service time \u4e3a {sjf['p99_service_time_s']:.1f}s\uff0c\u5b58\u5728\u5c11\u91cf\u957f\u5c3e\u8bf7\u6c42\u3002
- SAILP \u7684\u9996\u6b21\u7b49\u5f85\u4ec5 {sailp['waiting_time_mean']:.1f}s\uff0c\u4f46\u5e73\u5747 service time \u8fbe\u5230 {sailp['service_time_mean']:.1f}s\uff0cready queue peak \u4e3a {int(sailp['ready_queue_peak'])}\uff0cdependency stall \u4e3a {sailp['dependency_stall_mean']:.1f}s\u3002\u5b83\u5feb\u901f\u63a5\u7eb3\u8bf7\u6c42\uff0c\u5374\u5728 DAG \u5185\u90e8\u4ea7\u751f\u4e25\u91cd\u7b49\u5f85\u3002
- \u4e09\u79cd\u7b56\u7565\u7684 device busy \u6bd4\u4f8b\u7ea6 89%-95%\uff0c\u5e76\u884c\u5229\u7528\u7387\u7ea6 91.5%-91.7%\u3002\u6027\u80fd\u5dee\u5f02\u4e3b\u8981\u6765\u81ea\u6267\u884c\u987a\u5e8f\u3001\u961f\u5217\u79ef\u538b\u548c DAG \u5185\u90e8\u7b49\u5f85\uff0c\u800c\u4e0d\u662f GPU \u603b\u4f53\u7a7a\u95f2\u3002
- \u4e09\u6b21\u8fd0\u884c\u7684\u603b token \u6570\u5dee\u5f02\u7ea6 {token_spread:.2f}%\u3002\u5728\u8fd9\u7ec4\u914d\u7f6e\u4e0b\uff0cSJF \u662f\u6700\u597d\u7684\u7cfb\u7edf\u7ea7\u57fa\u7ebf\uff1bSAILP \u5f53\u524d\u5b9e\u73b0\u4e0d\u9002\u5408\u4f5c\u4e3a\u8bf7\u6c42\u5b8c\u6210\u65f6\u95f4\u57fa\u7ebf\u3002

## \u6570\u636e\u8fb9\u754c

- \u5360\u7528\u65f6\u95f4\u7ebf\u6765\u81ea\u7ed3\u679c JSON \u4e2d\u6bcf\u4e2a op \u7684\u5b9e\u9645 start/end \u65f6\u95f4\uff0c\u662f\u8c03\u5ea6\u5c42 device busy/idle\uff0c\u4e0d\u662f nvidia-smi SM utilization\u3002
- \u672c\u6b21\u5b8c\u6574\u8fd0\u884c\u6ca1\u6709\u8fde\u7eed\u4fdd\u5b58 GPU \u786c\u4ef6\u76d1\u63a7 CSV\uff0c\u56e0\u6b64\u4e0d\u7ed8\u5236\u663e\u5b58\u3001\u529f\u8017\u6216 SM utilization \u65f6\u5e8f\u3002
- \u5b8c\u6574\u6307\u6807\uff1aanalysis/scheduler_metrics.csv\u3002
- \u76f8\u5bf9\u6bd4\u8f83\uff1aanalysis/relative_vs_fcfs.csv\u3002
- Device \u6570\u636e\uff1aanalysis/device_busy_comparison.csv\u3002
- Dataset \u6570\u636e\uff1aanalysis/dataset_metrics.csv\u3002
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    return REPORT_PATH


def main() -> None:
    path = build_report()
    print(path)


if __name__ == "__main__":
    main()
