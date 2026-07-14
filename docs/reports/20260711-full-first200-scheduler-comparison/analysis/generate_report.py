#!/usr/bin/env python3
"""Generate comparison tables, charts, and report.md for the full first200 run."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch


RUN_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = RUN_ROOT / "analysis"
FIGURE_DIR = RUN_ROOT / "figures"
REPORT_PATH = RUN_ROOT / "report.md"

SCHEDULERS = ["fcfs", "sjf", "sailp"]
LABELS = {"fcfs": "FCFS", "sjf": "SJF", "sailp": "SAILP"}
COLORS = {"fcfs": "#4C78A8", "sjf": "#F58518", "sailp": "#54A24B"}
PHYSICAL_GPUS = {str(i): i + 3 for i in range(5)}
DATASET_ORDER = [
    "gsm8k",
    "strategyqa",
    "mmlu_pro",
    "math",
    "mbpp",
    "hotpotqa",
    "swebench_verified",
]


def configure_plot_style() -> None:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    font = next((name for name in candidates if name in installed), "DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": font,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#B8B8B8",
            "axes.grid": True,
            "grid.color": "#E4E4E4",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "font.size": 10,
        }
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def detail_path(scheduler: str) -> Path:
    matches = list((RUN_ROOT / scheduler).glob("*_run1.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one detail JSON for {scheduler}, found {matches}")
    return matches[0]


def summary_path(scheduler: str) -> Path:
    matches = list((RUN_ROOT / scheduler).glob("*_run1_summary.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one summary JSON for {scheduler}, found {matches}")
    return matches[0]


def percentile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.percentile(array, q)) if len(array) else math.nan


def request_tokens(request: dict[str, Any]) -> tuple[int, int, int]:
    input_tokens = 0
    output_tokens = 0
    for metrics in (request.get("op_metrics") or {}).values():
        input_tokens += int(metrics.get("input_tokens") or 0)
        output_tokens += int(metrics.get("output_tokens") or 0)
    return input_tokens, output_tokens, input_tokens + output_tokens


def merge_intervals(intervals: list[tuple[float, float]], gap_tolerance: float = 0.0) -> list[tuple[float, float]]:
    ordered = sorted((float(start), float(end)) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        current = merged[-1]
        if start <= current[1] + gap_tolerance:
            current[1] = max(current[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def extract_intervals(rows: list[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for request in rows:
        benchmark = request.get("benchmark") or {}
        assignments = request.get("worker_assignments") or {}
        for op, span in benchmark.items():
            if not isinstance(span, list) or len(span) < 2:
                continue
            worker = assignments.get(op)
            if worker is None:
                continue
            start, end = float(span[0]), float(span[1])
            if end > start:
                intervals[str(worker)].append((start, end))
    return {worker: merge_intervals(spans) for worker, spans in intervals.items()}


def format_seconds(value: float) -> str:
    if value >= 3600:
        return f"{value / 3600:.2f} h"
    if value >= 60:
        return f"{value / 60:.2f} min"
    return f"{value:.2f} s"


def percent_change(value: float, baseline: float) -> float:
    return (value / baseline - 1.0) * 100.0


def build_relative_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    fields = [
        ("makespan", "system_makespan", "lower"),
        ("total_token_throughput", "total_token_throughput", "higher"),
        ("output_token_throughput", "output_token_throughput", "higher"),
        ("waiting_time_mean", "average_waiting_time", "lower"),
        ("service_time_mean", "average_service_time", "lower"),
        ("avg_completion_time_s", "average_completion_time", "lower"),
        ("ready_queue_peak", "ready_queue_peak", "lower"),
        ("scheduler_overhead_seconds", "scheduler_overhead", "lower"),
        ("dependency_stall_mean", "dependency_stall", "lower"),
    ]
    baseline = metrics.loc["fcfs"]
    rows: list[dict[str, Any]] = []
    for field, label, direction in fields:
        rows.append(
            {
                "metric": label,
                "preferred_direction": direction,
                "sjf_vs_fcfs_pct": percent_change(float(metrics.loc["sjf", field]), float(baseline[field])),
                "sailp_vs_fcfs_pct": percent_change(
                    float(metrics.loc["sailp", field]), float(baseline[field])
                ),
            }
        )
    return pd.DataFrame(rows)


def add_bar_labels(ax: plt.Axes, values: list[float], fmt: str = "{:.1f}") -> None:
    for patch, value in zip(ax.patches, values):
        if not np.isfinite(value):
            continue
        ax.annotate(
            fmt.format(value),
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_device_occupancy(
    summaries: dict[str, dict[str, Any]],
    intervals: dict[str, dict[str, list[tuple[float, float]]]],
) -> None:
    max_minutes = max(float(summaries[s]["makespan"]) for s in SCHEDULERS) / 60.0
    expected_arrival_minutes = 1400 / 0.15 / 60.0
    fig, axes = plt.subplots(3, 1, figsize=(18, 11), sharex=True, constrained_layout=True)

    for ax, scheduler in zip(axes, SCHEDULERS):
        summary = summaries[scheduler]
        busy_pct = summary["device_busy_pct"]
        for row, worker in enumerate(["0", "1", "2", "3", "4"]):
            spans = [((start / 60.0), (end - start) / 60.0) for start, end in intervals[scheduler].get(worker, [])]
            ax.broken_barh(spans, (row - 0.34, 0.68), facecolors="#2E8B57", edgecolors="none", rasterized=True)
        labels = [
            f"GPU {PHYSICAL_GPUS[w]}  {float(busy_pct[w]) * 100:.1f}%"
            for w in ["0", "1", "2", "3", "4"]
        ]
        ax.set_yticks(range(5), labels)
        ax.set_ylim(-0.7, 4.7)
        ax.invert_yaxis()
        ax.set_xlim(0, max_minutes * 1.01)
        ax.axvline(expected_arrival_minutes, color="#555555", linestyle="--", linewidth=1.1)
        ax.set_title(
            f"{LABELS[scheduler]}  |  makespan {float(summary['makespan']) / 60:.1f} min  |  "
            f"average device busy {np.mean(list(busy_pct.values())) * 100:.1f}%",
            loc="left",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

    axes[-1].set_xlabel("Elapsed time (minutes)")
    fig.suptitle("Device occupancy comparison (green = at least one DAG op running)", fontsize=15, fontweight="bold")
    fig.legend(
        handles=[
            Patch(facecolor="#2E8B57", label="Running"),
            plt.Line2D([0], [0], color="#555555", linestyle="--", label="Expected end of Poisson arrival window"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=False,
    )
    fig.savefig(FIGURE_DIR / "device_occupancy_comparison.png", dpi=220)
    plt.close(fig)


def plot_core_metrics(metrics: pd.DataFrame) -> None:
    specs = [
        ("makespan_minutes", "System makespan", "minutes", False, "{:.1f}"),
        ("total_token_throughput", "Total throughput", "token/s", False, "{:.0f}"),
        ("output_token_throughput", "Output throughput", "token/s", False, "{:.1f}"),
        ("waiting_time_mean", "Average waiting time", "seconds (log scale)", True, "{:.1f}"),
        ("service_time_mean", "Average service time", "seconds (log scale)", True, "{:.1f}"),
        ("ready_queue_peak", "Ready queue peak", "requests", False, "{:.0f}"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    colors = [COLORS[s] for s in SCHEDULERS]
    labels = [LABELS[s] for s in SCHEDULERS]
    for ax, (field, title, ylabel, log_scale, fmt) in zip(axes.flat, specs):
        values = [float(metrics.loc[s, field]) for s in SCHEDULERS]
        ax.bar(labels, values, color=colors, width=0.62)
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        add_bar_labels(ax, values, fmt)
    fig.suptitle("Scheduler system metrics", fontsize=15, fontweight="bold")
    fig.savefig(FIGURE_DIR / "scheduler_metrics_comparison.png", dpi=220)
    plt.close(fig)


def plot_device_busy(device_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    x = np.arange(5)
    width = 0.24
    for index, scheduler in enumerate(SCHEDULERS):
        subset = device_df[device_df["scheduler"] == scheduler].sort_values("internal_device")
        values = subset["running_time_pct"].to_numpy(dtype=float)
        bars = ax.bar(x + (index - 1) * width, values, width, label=LABELS[scheduler], color=COLORS[scheduler])
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=2, fontsize=8)
    ax.set_xticks(x, [f"GPU {gpu}" for gpu in range(3, 8)])
    ax.set_ylim(80, 100)
    ax.set_ylabel("Running time (%)")
    ax.set_title("Per-device running-time ratio (axis starts at 80%)", fontweight="bold")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIGURE_DIR / "device_busy_comparison.png", dpi=220)
    plt.close(fig)


def plot_dataset_service(dataset_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(15, 7), constrained_layout=True)
    x = np.arange(len(DATASET_ORDER))
    width = 0.24
    for index, scheduler in enumerate(SCHEDULERS):
        subset = dataset_df[dataset_df["scheduler"] == scheduler].set_index("dataset").reindex(DATASET_ORDER)
        values = subset["avg_service_time_s"].to_numpy(dtype=float)
        ax.bar(x + (index - 1) * width, values, width, label=LABELS[scheduler], color=COLORS[scheduler])
    ax.set_xticks(x, DATASET_ORDER, rotation=25, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Average service time (seconds, log scale)")
    ax.set_title("Average service time by dataset", fontweight="bold")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", which="both")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIGURE_DIR / "dataset_service_time_comparison.png", dpi=220)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    summaries: dict[str, dict[str, Any]],
    metrics: pd.DataFrame,
    device_df: pd.DataFrame,
    dataset_df: pd.DataFrame,
    relative_df: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    first = summaries["fcfs"]
    run_info = first["run_info"]
    fcfs = metrics.loc["fcfs"]
    sjf = metrics.loc["sjf"]
    sailp = metrics.loc["sailp"]

    core_headers = [
        "策略",
        "完成",
        "系统总完成时间",
        "总 tokens",
        "输出 tokens",
        "总吞吐量",
        "输出吞吐量",
        "平均等待",
        "平均 service time",
    ]
    core_rows: list[list[str]] = []
    for scheduler in SCHEDULERS:
        row = metrics.loc[scheduler]
        core_rows.append(
            [
                LABELS[scheduler],
                f"{int(row['completed'])}/{int(row['count'])}",
                f"{row['makespan']:.1f}s ({row['makespan_minutes']:.1f}min)",
                f"{int(row['total_tokens']):,}",
                f"{int(row['output_tokens']):,}",
                f"{row['total_token_throughput']:.1f} token/s",
                f"{row['output_token_throughput']:.1f} token/s",
                f"{row['waiting_time_mean']:.1f}s",
                f"{row['service_time_mean']:.1f}s",
            ]
        )

    scheduling_headers = [
        "策略",
        "平均完成时间",
        "P95 service time",
        "ready queue avg / peak",
        "dependency stall",
        "调度开销",
        "负载不均衡",
        "并行利用率",
    ]
    scheduling_rows: list[list[str]] = []
    for scheduler in SCHEDULERS:
        row = metrics.loc[scheduler]
        scheduling_rows.append(
            [
                LABELS[scheduler],
                f"{row['avg_completion_time_s']:.1f}s",
                f"{row['p95_service_time_s']:.1f}s",
                f"{row['ready_queue_avg']:.1f} / {int(row['ready_queue_peak'])}",
                f"{row['dependency_stall_mean']:.1f}s",
                f"{row['scheduler_overhead_seconds']:.1f}s ({row['scheduler_overhead_pct'] * 100:.2f}%)",
                f"{row['load_imbalance']:.4f}",
                f"{row['parallelism_utilization'] * 100:.2f}%",
            ]
        )

    relative_names = {
        "system_makespan": "系统总完成时间",
        "total_token_throughput": "总 token 吞吐量",
        "output_token_throughput": "输出 token 吞吐量",
        "average_waiting_time": "平均等待时间",
        "average_service_time": "平均 service time",
        "average_completion_time": "平均完成时间",
        "ready_queue_peak": "ready queue peak",
        "scheduler_overhead": "调度开销",
        "dependency_stall": "dependency stall",
    }
    relative_headers = ["指标", "优选方向", "SJF vs FCFS", "SAILP vs FCFS"]
    relative_rows = [
        [
            relative_names[str(row.metric)],
            "越低越好" if row.preferred_direction == "lower" else "越高越好",
            f"{float(row.sjf_vs_fcfs_pct):+.2f}%",
            f"{float(row.sailp_vs_fcfs_pct):+.2f}%",
        ]
        for row in relative_df.itertuples(index=False)
    ]

    device_headers = ["Physical GPU", "FCFS", "SJF", "SAILP"]
    device_rows: list[list[str]] = []
    for gpu in range(3, 8):
        values = []
        for scheduler in SCHEDULERS:
            value = device_df[(device_df["scheduler"] == scheduler) & (device_df["physical_gpu"] == gpu)][
                "running_time_pct"
            ].iloc[0]
            values.append(f"{value:.2f}%")
        device_rows.append([str(gpu), *values])

    dataset_headers = ["Dataset", "FCFS", "SJF", "SAILP"]
    dataset_rows: list[list[str]] = []
    for dataset in DATASET_ORDER:
        values = []
        for scheduler in SCHEDULERS:
            value = dataset_df[
                (dataset_df["scheduler"] == scheduler) & (dataset_df["dataset"] == dataset)
            ]["avg_service_time_s"].iloc[0]
            values.append(f"{value:.1f}s")
        dataset_rows.append([dataset, *values])

    total_token_spread = (
        (metrics["total_tokens"].max() - metrics["total_tokens"].min()) / metrics["total_tokens"].mean() * 100
    )
    report = f"""# Full first200 调度策略系统性能报告

## 实验配置

| 项目 | 配置 |
| --- | --- |
| 数据规模 | 7 个 dataset，每个随机选取 200 条，共 1,400 requests |
| 数据顺序 | 长度过滤后按 seed `{manifest['seed']}` 随机抽样，mixed workload 全局随机打乱 |
| 数据集 | {', '.join(manifest['datasets'])} |
| 长度过滤 | start prompt `< 14000` tokens；start prompt + op max tokens `<= 15500` |
| 模型 | `{run_info['model_path']}` |
| 推理后端 | real vLLM `{run_info['packages']['vllm']}`，CUDA / NVIDIA A800 |
| GPU | physical GPU `3,4,5,6,7`，共 5 张 |
| max model len | `{run_info['max_model_len']}` |
| output max tokens | `{first['output_max_tokens']}` per DAG op |
| GPU memory utilization | `{run_info['gpu_memory_utilization']}` |
| 到达过程 | Poisson，`0.15 req/s`，arrival batch size `1`，seed `{first['arrival_seed']}` |
| 调度策略 | FCFS、SJF、SAILP |
| 代码版本 | `{run_info['git_commit']}` |

## 核心结果

{markdown_table(core_headers, core_rows)}

![Scheduler metrics](figures/scheduler_metrics_comparison.png)

## Device 占用时间线

绿色表示该 device 正在执行至少一个 DAG op；空白表示没有 op 运行。内部 device `0-4` 映射到 physical GPU `3-7`。虚线表示按 `1400 / 0.15` 计算的理论泊松到达窗口末端。

![Device occupancy comparison](figures/device_occupancy_comparison.png)

{markdown_table(device_headers, device_rows)}

![Device busy comparison](figures/device_busy_comparison.png)

## 调度与队列指标

{markdown_table(scheduling_headers, scheduling_rows)}

`平均等待`只表示 request 到达后到第一个 op 开始的时间；`平均 service time`表示第一个 op 开始到整个 DAG 完成的时间。二者必须同时观察。

## 相对 FCFS 的变化

{markdown_table(relative_headers, relative_rows)}

百分比为 `(策略值 / FCFS - 1) x 100%`。时间、队列和 stall 指标通常越低越好；吞吐量越高越好。

## Dataset 平均 Service Time

{markdown_table(dataset_headers, dataset_rows)}

![Dataset service time comparison](figures/dataset_service_time_comparison.png)

## 结论

- 三种策略均完成 `1400/1400`，成功率 100%，没有 context length、OOM 或运行崩溃。
- **SJF 的系统总完成时间最短**：`{sjf['makespan']:.1f}s`，比 FCFS 缩短 `{abs(percent_change(sjf['makespan'], fcfs['makespan'])):.2f}%`；总吞吐量比 FCFS 提高 `{percent_change(sjf['total_token_throughput'], fcfs['total_token_throughput']):.2f}%`。
- FCFS 的平均等待时间较高（`{fcfs['waiting_time_mean']:.1f}s`），但请求开始执行后的平均 service time 最短（`{fcfs['service_time_mean']:.1f}s`）。
- SJF 将平均等待降低到 `{sjf['waiting_time_mean']:.1f}s`，平均完成时间为 `{sjf['avg_completion_time_s']:.1f}s`；但存在少量长尾请求，因此平均 service time 高于 FCFS。
- SAILP 的首次等待仅 `{sailp['waiting_time_mean']:.1f}s`，但平均 service time 达到 `{sailp['service_time_mean']:.1f}s`，ready queue peak 为 `{int(sailp['ready_queue_peak'])}`，dependency stall 为 `{sailp['dependency_stall_mean']:.1f}s`。这说明它快速接纳大量请求，却在 DAG 内部产生严重等待，不能把低 `avg_wait_s` 解释为低请求完成时间。
- 三种策略的 device busy 比例和并行利用率非常接近（约 89%-95% 和 91.5%-91.7%）。性能差异主要来自执行顺序、队列积压和 DAG 内部等待，而不是 GPU 总体空闲。
- 三次运行的总 token 数差异约 `{total_token_spread:.2f}%`，系统吞吐量比较具有可比性；其中 SJF 当前是这组配置下最好的系统级基线。

## 数据边界

- 占用时间线来自结果 JSON 中每个 op 的实际 start/end 时间，是调度层 device busy/idle，不是 `nvidia-smi` SM utilization。
- 本次完整运行没有连续保存 GPU 硬件监控 CSV，因此报告不绘制显存、功耗或 SM utilization 时序。
- 原始结果、汇总表和可复现分析脚本均保存在本目录；核心 CSV 位于 `analysis/`，图表位于 `figures/`。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    configure_plot_style()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_json(RUN_ROOT / "manifest_medium_first200.json")
    summaries: dict[str, dict[str, Any]] = {}
    details: dict[str, list[dict[str, Any]]] = {}
    all_intervals: dict[str, dict[str, list[tuple[float, float]]]] = {}
    metrics_rows: list[dict[str, Any]] = []
    device_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []

    for scheduler in SCHEDULERS:
        summary = load_json(summary_path(scheduler))
        rows = load_json(detail_path(scheduler))
        if len(rows) != 1400 or summary.get("completed") != 1400 or summary.get("success_rate") != 1.0:
            raise RuntimeError(f"Incomplete result for {scheduler}: rows={len(rows)} summary={summary}")
        if any(request.get("status") != "completed" for request in rows):
            raise RuntimeError(f"Non-completed request found for {scheduler}")

        summaries[scheduler] = summary
        details[scheduler] = rows
        merged = extract_intervals(rows)
        all_intervals[scheduler] = merged

        service_times = [float(row["service_time"]) for row in rows]
        completion_times = [float(row["latency"]) for row in rows]
        metrics_rows.append(
            {
                "scheduler": scheduler,
                "count": summary["count"],
                "completed": summary["completed"],
                "success_rate": summary["success_rate"],
                "makespan": summary["makespan"],
                "makespan_minutes": float(summary["makespan"]) / 60.0,
                "input_tokens": summary["input_tokens"],
                "output_tokens": summary["output_tokens"],
                "total_tokens": summary["total_tokens"],
                "request_throughput": summary["request_throughput"],
                "input_token_throughput": summary["input_token_throughput"],
                "output_token_throughput": summary["output_token_throughput"],
                "total_token_throughput": summary["total_token_throughput"],
                "waiting_time_mean": summary["waiting_time_mean"],
                "service_time_mean": summary["service_time_mean"],
                "p50_service_time_s": percentile(service_times, 50),
                "p95_service_time_s": percentile(service_times, 95),
                "p99_service_time_s": percentile(service_times, 99),
                "avg_completion_time_s": float(np.mean(completion_times)),
                "scheduler_overhead_seconds": summary["scheduler_overhead_seconds"],
                "scheduler_overhead_pct": summary["scheduler_overhead_pct"],
                "ready_queue_avg": summary["ready_queue_avg"],
                "ready_queue_peak": summary["ready_queue_peak"],
                "critical_path_mean": summary["critical_path_mean"],
                "dag_parallelism_mean": summary["dag_parallelism_mean"],
                "dependency_stall_mean": summary["dependency_stall_mean"],
                "cross_device_dependencies_mean": summary["cross_device_dependencies_mean"],
                "load_imbalance": summary["load_imbalance"],
                "parallelism_utilization": summary["parallelism_utilization"],
            }
        )

        for worker in ["0", "1", "2", "3", "4"]:
            summary_busy = float(summary["device_busy_seconds"][worker])
            union_busy = sum(end - start for start, end in merged.get(worker, []))
            device_rows.append(
                {
                    "scheduler": scheduler,
                    "internal_device": int(worker),
                    "physical_gpu": PHYSICAL_GPUS[worker],
                    "busy_seconds": summary_busy,
                    "union_busy_seconds": union_busy,
                    "running_time_pct": float(summary["device_busy_pct"][worker]) * 100.0,
                    "output_tokens_per_second": float(summary["device_output_tokens_per_second"][worker]),
                }
            )
            for start, end in merged.get(worker, []):
                interval_rows.append(
                    {
                        "scheduler": scheduler,
                        "internal_device": int(worker),
                        "physical_gpu": PHYSICAL_GPUS[worker],
                        "start_time_s": start,
                        "end_time_s": end,
                        "duration_s": end - start,
                    }
                )

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for request in rows:
            grouped[str(request["dataset"])].append(request)
        for dataset in DATASET_ORDER:
            dataset_requests = grouped[dataset]
            token_rows = [request_tokens(request) for request in dataset_requests]
            service = [float(request["service_time"]) for request in dataset_requests]
            waiting = [float(request["idle_time"]) for request in dataset_requests]
            completion = [float(request["latency"]) for request in dataset_requests]
            dataset_rows.append(
                {
                    "scheduler": scheduler,
                    "dataset": dataset,
                    "count": len(dataset_requests),
                    "avg_waiting_time_s": float(np.mean(waiting)),
                    "avg_service_time_s": float(np.mean(service)),
                    "p95_service_time_s": percentile(service, 95),
                    "avg_completion_time_s": float(np.mean(completion)),
                    "input_tokens": sum(value[0] for value in token_rows),
                    "output_tokens": sum(value[1] for value in token_rows),
                    "total_tokens": sum(value[2] for value in token_rows),
                }
            )

    metrics_df = pd.DataFrame(metrics_rows).set_index("scheduler")
    device_df = pd.DataFrame(device_rows)
    dataset_df = pd.DataFrame(dataset_rows)
    intervals_df = pd.DataFrame(interval_rows)
    relative_df = build_relative_comparison(metrics_df)

    metrics_df.to_csv(ANALYSIS_DIR / "scheduler_metrics.csv", encoding="utf-8-sig")
    device_df.to_csv(ANALYSIS_DIR / "device_busy_comparison.csv", index=False, encoding="utf-8-sig")
    dataset_df.to_csv(ANALYSIS_DIR / "dataset_metrics.csv", index=False, encoding="utf-8-sig")
    intervals_df.to_csv(ANALYSIS_DIR / "device_intervals_merged.csv", index=False, encoding="utf-8-sig")
    relative_df.to_csv(ANALYSIS_DIR / "relative_vs_fcfs.csv", index=False, encoding="utf-8-sig")

    plot_device_occupancy(summaries, all_intervals)
    plot_core_metrics(metrics_df)
    plot_device_busy(device_df)
    plot_dataset_service(dataset_df)
    from build_report import build_report

    build_report(RUN_ROOT)

    print(f"report={REPORT_PATH}")
    print(f"figures={FIGURE_DIR}")
    print(f"metrics={ANALYSIS_DIR / 'scheduler_metrics.csv'}")


if __name__ == "__main__":
    main()
