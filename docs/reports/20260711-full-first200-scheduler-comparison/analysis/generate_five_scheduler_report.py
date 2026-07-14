#!/usr/bin/env python3
"""Generate the five-scheduler first200 comparison report and figures."""

from __future__ import annotations

import json
import math
import re
import shutil
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch


REPORT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_RUNS_ROOT = REPORT_ROOT.parent
NEW_RUN_ROOT = REMOTE_RUNS_ROOT / (
    "20260713-160000-full-first200-poisson015-rhsail-darc-"
    "vllm5-max32768-out2048-mem075"
)
ANALYSIS_DIR = REPORT_ROOT / "analysis"
FIGURE_DIR = REPORT_ROOT / "figures"
REPORT_PATH = REPORT_ROOT / "report.md"
PUBLISH_ROOT = (
    REMOTE_RUNS_ROOT.parent
    / "mfe-ascend"
    / "docs"
    / "reports"
    / "20260711-full-first200-scheduler-comparison"
)

SCHEDULERS = ["fcfs", "sjf", "sailp", "rhsail", "darc"]
LABELS = {
    "fcfs": "FCFS",
    "sjf": "SJF",
    "sailp": "SAILP",
    "rhsail": "RH-SAIL",
    "darc": "DARC",
}
COLORS = {
    "fcfs": "#4C78A8",
    "sjf": "#F58518",
    "sailp": "#D45087",
    "rhsail": "#7A8F3A",
    "darc": "#C9A227",
}
PHYSICAL_GPUS = {str(index): index + 3 for index in range(5)}
DATASET_ORDER = [
    "gsm8k",
    "strategyqa",
    "mmlu_pro",
    "math",
    "mbpp",
    "hotpotqa",
    "swebench_verified",
]
EXPECTED_REQUESTS = 1400
POISSON_RATE = 0.15
TZ_UTC8 = timezone(timedelta(hours=8))


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
            "savefig.bbox": "tight",
        }
    )


def scheduler_root(scheduler: str) -> Path:
    root = NEW_RUN_ROOT if scheduler in {"rhsail", "darc"} else REPORT_ROOT
    return root / scheduler


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def only_match(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {root / pattern}, found {matches}")
    return matches[0]


def detail_path(scheduler: str) -> Path:
    return only_match(scheduler_root(scheduler), "*_run1.json")


def summary_path(scheduler: str) -> Path:
    return only_match(scheduler_root(scheduler), "*_run1_summary.json")


def percentile(values: Iterable[float], q: float) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.percentile(array, q)) if len(array) else math.nan


def percent_change(value: float, baseline: float) -> float:
    return (value / baseline - 1.0) * 100.0


def request_tokens(request: dict[str, Any]) -> tuple[int, int, int]:
    input_tokens = 0
    output_tokens = 0
    for metrics in (request.get("op_metrics") or {}).values():
        input_tokens += int(metrics.get("input_tokens") or 0)
        output_tokens += int(metrics.get("output_tokens") or 0)
    return input_tokens, output_tokens, input_tokens + output_tokens


def merge_intervals(
    intervals: list[tuple[float, float]], gap_tolerance: float = 0.0
) -> list[tuple[float, float]]:
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


def extract_device_intervals(
    rows: list[dict[str, Any]],
) -> dict[str, list[tuple[float, float]]]:
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


def request_continuity(request: dict[str, Any]) -> dict[str, float]:
    spans = []
    for span in (request.get("benchmark") or {}).values():
        if isinstance(span, list) and len(span) >= 2 and float(span[1]) > float(span[0]):
            spans.append((float(span[0]), float(span[1])))
    merged = merge_intervals(spans)
    active_wall = sum(end - start for start, end in merged)
    gaps = [merged[index + 1][0] - merged[index][1] for index in range(len(merged) - 1)]
    service = float(request.get("service_time") or 0.0)
    dormant = max(0.0, service - active_wall)
    return {
        "active_wall_time_s": active_wall,
        "dormant_time_s": dormant,
        "dormant_fraction": dormant / service if service > 0 else 0.0,
        "max_inter_op_gap_s": max(gaps, default=0.0),
        "service_stretch": service / active_wall if active_wall > 0 else math.nan,
    }


def parse_runner_intervals() -> dict[str, tuple[float, float]]:
    text = (NEW_RUN_ROOT / "runner.log").read_text(encoding="utf-8")
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}
    pattern = re.compile(
        r"^===== (START|END) (rhsail|darc) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) =====$",
        re.MULTILINE,
    )
    for kind, scheduler, timestamp in pattern.findall(text):
        epoch = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ_UTC8).timestamp()
        (starts if kind == "START" else ends)[scheduler] = epoch
    missing = [scheduler for scheduler in ("rhsail", "darc") if scheduler not in starts or scheduler not in ends]
    if missing:
        raise RuntimeError(f"Incomplete runner intervals for {missing}: {text}")
    return {scheduler: (starts[scheduler], ends[scheduler]) for scheduler in starts}


def load_gpu_hardware_summary() -> pd.DataFrame:
    csv_path = NEW_RUN_ROOT / "gpu_metrics.csv"
    intervals = parse_runner_intervals()
    frame = pd.read_csv(csv_path)
    frame = frame[frame["gpu_index"].between(3, 7)].copy()
    rows: list[dict[str, Any]] = []
    for scheduler, (start, end) in intervals.items():
        subset = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)]
        if subset.empty:
            raise RuntimeError(f"No GPU samples for {scheduler}")
        for gpu, group in subset.groupby("gpu_index"):
            rows.append(
                {
                    "scheduler": scheduler,
                    "physical_gpu": int(gpu),
                    "samples": len(group),
                    "avg_gpu_util_pct": float(group["utilization_gpu_pct"].mean()),
                    "p95_gpu_util_pct": percentile(group["utilization_gpu_pct"], 95),
                    "avg_memory_mib": float(group["memory_used_mb"].mean()),
                    "max_memory_mib": float(group["memory_used_mb"].max()),
                    "avg_power_w": float(group["power_w"].mean()),
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
            fontsize=7,
        )


def plot_core_metrics(metrics: pd.DataFrame) -> None:
    specs = [
        ("makespan_minutes", "System makespan", "minutes", False, "{:.1f}"),
        ("total_token_throughput", "Total throughput", "token/s", False, "{:.0f}"),
        ("output_token_throughput", "Output throughput", "token/s", False, "{:.1f}"),
        ("waiting_time_mean", "Average waiting time", "seconds (log scale)", True, "{:.1f}"),
        ("service_time_mean", "Average service time", "seconds (log scale)", True, "{:.1f}"),
        ("avg_completion_time_s", "Average completion time", "seconds (log scale)", True, "{:.1f}"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    labels = [LABELS[scheduler] for scheduler in SCHEDULERS]
    colors = [COLORS[scheduler] for scheduler in SCHEDULERS]
    for ax, (field, title, ylabel, log_scale, fmt) in zip(axes.flat, specs):
        values = [float(metrics.loc[scheduler, field]) for scheduler in SCHEDULERS]
        ax.bar(labels, values, color=colors, width=0.68)
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        add_bar_labels(ax, values, fmt)
    fig.suptitle("Five-scheduler system performance", fontsize=15, fontweight="bold")
    fig.savefig(FIGURE_DIR / "scheduler_metrics_comparison.png", dpi=220)
    plt.close(fig)


def plot_continuity(metrics: pd.DataFrame, continuity: pd.DataFrame) -> None:
    specs = [
        (metrics, "p95_service_time_s", "P95 service time", "seconds (log scale)", True),
        (continuity, "p95_max_inter_op_gap_s", "P95 max inter-op gap", "seconds (log scale)", True),
        (continuity, "avg_dormant_fraction", "Average dormant fraction", "% of service time", False),
        (metrics, "scheduler_overhead_pct", "Scheduler overhead", "% of makespan", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    labels = [LABELS[scheduler] for scheduler in SCHEDULERS]
    colors = [COLORS[scheduler] for scheduler in SCHEDULERS]
    for ax, (frame, field, title, ylabel, log_scale) in zip(axes.flat, specs):
        values = [float(frame.loc[scheduler, field]) for scheduler in SCHEDULERS]
        if field in {"avg_dormant_fraction", "scheduler_overhead_pct"}:
            values = [value * 100.0 for value in values]
        plotted = [max(value, 1e-3) if log_scale else value for value in values]
        ax.bar(labels, plotted, color=colors, width=0.68)
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", which="both")
        ax.grid(axis="x", visible=False)
        add_bar_labels(ax, values, "{:.1f}")
    fig.suptitle("Request continuity and scheduling cost", fontsize=15, fontweight="bold")
    fig.savefig(FIGURE_DIR / "request_continuity_comparison.png", dpi=220)
    plt.close(fig)


def plot_device_occupancy(
    summaries: dict[str, dict[str, Any]],
    intervals: dict[str, dict[str, list[tuple[float, float]]]],
    details: dict[str, list[dict[str, Any]]],
) -> None:
    max_minutes = max(float(summaries[scheduler]["makespan"]) for scheduler in SCHEDULERS) / 60.0
    expected_arrival_minutes = EXPECTED_REQUESTS / POISSON_RATE / 60.0
    fig, axes = plt.subplots(
        len(SCHEDULERS),
        1,
        figsize=(18, 17),
        sharex=True,
        constrained_layout=True,
    )
    for ax, scheduler in zip(axes, SCHEDULERS):
        summary = summaries[scheduler]
        busy_pct = summary["device_busy_pct"]
        arrival_minutes = sorted(
            float(request["arrive_time"]) / 60.0 for request in details[scheduler]
        )
        ax.eventplot(
            arrival_minutes,
            orientation="horizontal",
            lineoffsets=0,
            linelengths=0.68,
            linewidths=0.42,
            colors="#343A40",
        )
        for row, worker in enumerate(["0", "1", "2", "3", "4"], start=1):
            spans = [
                (start / 60.0, (end - start) / 60.0)
                for start, end in intervals[scheduler].get(worker, [])
            ]
            ax.broken_barh(
                spans,
                (row - 0.34, 0.68),
                facecolors="#2E8B57",
                edgecolors="none",
                rasterized=True,
            )
        labels = ["Query arrivals"] + [
            f"GPU {PHYSICAL_GPUS[worker]}  {float(busy_pct[worker]) * 100:.1f}%"
            for worker in ["0", "1", "2", "3", "4"]
        ]
        ax.set_yticks(range(6), labels)
        ax.set_ylim(-0.7, 5.7)
        ax.invert_yaxis()
        ax.set_xlim(0, max_minutes * 1.01)
        ax.axvline(expected_arrival_minutes, color="#555555", linestyle="--", linewidth=1.1)
        ax.set_title(
            f"{LABELS[scheduler]} | makespan {float(summary['makespan']) / 60:.1f} min | "
            f"average device busy {np.mean(list(busy_pct.values())) * 100:.1f}%",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    axes[-1].set_xlabel("Elapsed time (minutes)")
    fig.suptitle(
        "Device occupancy comparison (green = at least one DAG op running)",
        fontsize=15,
        fontweight="bold",
    )
    fig.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                color="#343A40",
                marker="|",
                linestyle="none",
                markersize=10,
                label="Actual query arrival",
            ),
            Patch(facecolor="#2E8B57", label="Running"),
            plt.Line2D(
                [0],
                [0],
                color="#555555",
                linestyle="--",
                label="Expected end of Poisson arrival window",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        frameon=False,
    )
    fig.savefig(FIGURE_DIR / "device_occupancy_comparison.png", dpi=220)
    plt.close(fig)


def plot_device_busy(device: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 7), constrained_layout=True)
    x = np.arange(5)
    width = 0.15
    offsets = (np.arange(len(SCHEDULERS)) - (len(SCHEDULERS) - 1) / 2.0) * width
    all_values: list[float] = []
    for offset, scheduler in zip(offsets, SCHEDULERS):
        subset = device[device["scheduler"] == scheduler].sort_values("internal_device")
        values = subset["running_time_pct"].to_numpy(dtype=float)
        all_values.extend(values.tolist())
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=LABELS[scheduler],
            color=COLORS[scheduler],
        )
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=2, fontsize=6.5)
    ax.set_xticks(x, [f"GPU {gpu}" for gpu in range(3, 8)])
    lower = max(0.0, math.floor((min(all_values) - 5.0) / 5.0) * 5.0)
    ax.set_ylim(lower, 100)
    ax.set_ylabel("Running time (%)")
    ax.set_title(f"Per-device running-time ratio (axis starts at {lower:.0f}%)", fontweight="bold")
    ax.legend(frameon=False, ncol=5)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIGURE_DIR / "device_busy_comparison.png", dpi=220)
    plt.close(fig)


def plot_dataset_service(dataset: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
    x = np.arange(len(DATASET_ORDER))
    width = 0.15
    offsets = (np.arange(len(SCHEDULERS)) - (len(SCHEDULERS) - 1) / 2.0) * width
    for offset, scheduler in zip(offsets, SCHEDULERS):
        subset = dataset[dataset["scheduler"] == scheduler].set_index("dataset").reindex(DATASET_ORDER)
        values = subset["avg_service_time_s"].to_numpy(dtype=float)
        ax.bar(
            x + offset,
            values,
            width,
            label=LABELS[scheduler],
            color=COLORS[scheduler],
        )
    ax.set_xticks(x, DATASET_ORDER, rotation=25, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Average service time (seconds, log scale)")
    ax.set_title("Average service time by dataset", fontweight="bold")
    ax.legend(frameon=False, ncol=5)
    ax.grid(axis="y", which="both")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIGURE_DIR / "dataset_service_time_comparison.png", dpi=220)
    plt.close(fig)


def plot_gpu_hardware(hardware: pd.DataFrame) -> None:
    new_schedulers = ["rhsail", "darc"]
    specs = [
        ("avg_gpu_util_pct", "Average GPU utilization", "%"),
        ("max_memory_mib", "Maximum GPU memory", "GiB"),
        ("avg_power_w", "Average GPU power", "W"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    x = np.arange(5)
    width = 0.34
    for ax, (field, title, ylabel) in zip(axes, specs):
        for index, scheduler in enumerate(new_schedulers):
            subset = hardware[hardware["scheduler"] == scheduler].sort_values("physical_gpu")
            values = subset[field].to_numpy(dtype=float)
            if field == "max_memory_mib":
                values = values / 1024.0
            bars = ax.bar(
                x + (index - 0.5) * width,
                values,
                width,
                label=LABELS[scheduler],
                color=COLORS[scheduler],
            )
            ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=2, fontsize=7)
        ax.set_xticks(x, [f"GPU {gpu}" for gpu in range(3, 8)], rotation=20)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Hardware telemetry for the two new schedulers", fontsize=15, fontweight="bold")
    fig.savefig(FIGURE_DIR / "new_scheduler_gpu_hardware.png", dpi=220)
    plt.close(fig)


def markdown_table(headers: list[str], rows: Iterable[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def scheduler_columns(values: dict[str, str]) -> list[str]:
    return [values[scheduler] for scheduler in SCHEDULERS]


def build_report(
    summaries: dict[str, dict[str, Any]],
    metrics: pd.DataFrame,
    continuity: pd.DataFrame,
    device: pd.DataFrame,
    dataset: pd.DataFrame,
    relative: pd.DataFrame,
    hardware: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    old_info = summaries["fcfs"]["run_info"]
    new_info = summaries["rhsail"]["run_info"]

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

    continuity_rows = []
    for scheduler in SCHEDULERS:
        row = continuity.loc[scheduler]
        continuity_rows.append(
            [
                LABELS[scheduler],
                f"{row['avg_active_wall_time_s']:.1f}s",
                f"{row['avg_dormant_time_s']:.1f}s",
                f"{row['avg_dormant_fraction'] * 100:.1f}%",
                f"{row['p95_max_inter_op_gap_s']:.1f}s",
                f"{row['p95_service_stretch']:.2f}x",
            ]
        )

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
                f"{row['parallelism_utilization'] * 100:.2f}%",
            ]
        )

    relative_headers = ["指标", "优选方向"] + [
        f"{LABELS[scheduler]} vs FCFS" for scheduler in SCHEDULERS if scheduler != "fcfs"
    ]
    relative_rows = []
    for metric_name, direction in [
        ("system_makespan", "越低越好"),
        ("total_token_throughput", "越高越好"),
        ("output_token_throughput", "越高越好"),
        ("average_waiting_time", "越低越好"),
        ("average_service_time", "越低越好"),
        ("average_completion_time", "越低越好"),
        ("p95_service_time", "越低越好"),
        ("ready_queue_peak", "越低越好"),
        ("dependency_stall", "越低越好"),
        ("scheduler_overhead", "越低越好"),
    ]:
        subset = relative[relative["metric"] == metric_name].set_index("scheduler")
        relative_rows.append(
            [metric_name, direction]
            + [f"{subset.loc[scheduler, 'vs_fcfs_pct']:+.2f}%" for scheduler in SCHEDULERS[1:]]
        )

    device_headers = ["Physical GPU"] + [LABELS[scheduler] for scheduler in SCHEDULERS]
    device_rows = []
    for gpu in range(3, 8):
        values = {}
        for scheduler in SCHEDULERS:
            value = device[
                (device["scheduler"] == scheduler) & (device["physical_gpu"] == gpu)
            ]["running_time_pct"].iloc[0]
            values[scheduler] = f"{value:.2f}%"
        device_rows.append([str(gpu), *scheduler_columns(values)])

    dataset_headers = ["Dataset"] + [LABELS[scheduler] for scheduler in SCHEDULERS]
    dataset_rows = []
    for name in DATASET_ORDER:
        values = {}
        for scheduler in SCHEDULERS:
            value = dataset[
                (dataset["scheduler"] == scheduler) & (dataset["dataset"] == name)
            ]["avg_service_time_s"].iloc[0]
            values[scheduler] = f"{value:.1f}s"
        dataset_rows.append([name, *scheduler_columns(values)])

    hardware_rows = []
    for scheduler in ("rhsail", "darc"):
        subset = hardware[hardware["scheduler"] == scheduler]
        hardware_rows.append(
            [
                LABELS[scheduler],
                f"{subset['avg_gpu_util_pct'].mean():.1f}%",
                f"{subset['p95_gpu_util_pct'].mean():.1f}%",
                f"{subset['max_memory_mib'].max() / 1024.0:.2f} GiB",
                f"{subset['avg_power_w'].mean():.1f} W",
            ]
        )

    makespan_best = metrics["makespan"].idxmin()
    throughput_best = metrics["total_token_throughput"].idxmax()
    completion_best = metrics["avg_completion_time_s"].idxmin()
    rhsail = metrics.loc["rhsail"]
    sailp = metrics.loc["sailp"]
    darc = metrics.loc["darc"]
    fcfs = metrics.loc["fcfs"]
    token_spread = (
        (metrics["total_tokens"].max() - metrics["total_tokens"].min())
        / metrics["total_tokens"].mean()
        * 100.0
    )

    report = f"""# Full first200 五种调度策略系统性能报告

## 实验配置

| 项目 | 配置 |
| --- | --- |
| 数据规模 | 7 个 dataset，每个随机选取 200 条，共 1,400 requests |
| 数据顺序 | 长度过滤后按 seed `{manifest['seed']}` 随机抽样，mixed workload 全局随机打乱；五种策略使用同一 questions 文件 |
| 数据集 | {', '.join(manifest['datasets'])} |
| 长度过滤 | start prompt `< 14000` tokens；start prompt + op max tokens `<= 15500` |
| 模型 | `{new_info['model_path']}` |
| 推理后端 | real vLLM `{new_info['packages']['vllm']}`，CUDA / NVIDIA A800 |
| GPU | physical GPU `3,4,5,6,7`，共 5 张 |
| max model len | `{new_info['max_model_len']}` |
| output max tokens | `{summaries['rhsail']['output_max_tokens']}` per DAG op |
| GPU memory utilization | `{new_info['gpu_memory_utilization']}` |
| 到达过程 | Poisson `0.15 req/s`，arrival batch size `1`，seed `{summaries['rhsail']['arrival_seed']}` |
| 调度策略 | FCFS、SJF、SAILP、RH-SAIL、DARC |
| 代码版本 | FCFS/SJF/SAILP: `{old_info['git_commit']}`；RH-SAIL/DARC: `{new_info['git_commit']}` |

## 核心系统指标

{markdown_table(['策略', '完成', '系统总完成时间', '总 tokens', '输出 tokens', '请求吞吐量', '总吞吐量', '输出吞吐量'], core_rows)}

系统总完成时间和 token 吞吐量决定整批实验的系统效率；等待、service time 和平均完成时间用于判断效率是否以请求长尾为代价。

![Scheduler metrics](figures/scheduler_metrics_comparison.png)

## 请求时间指标

{markdown_table(['策略', '平均等待', '平均 service time', 'P50 service', 'P95 service', 'P99 service', '平均完成时间'], timing_rows)}

`平均等待`是请求到达至首个 op 开始；`service time`是首个 op 开始至整个 DAG 完成；`平均完成时间 = 平均等待 + 平均 service time`。

## 请求连续性

{markdown_table(['策略', '平均 active wall time', '平均 dormant time', '平均 dormant fraction', 'P95 最大算子间空档', 'P95 service stretch'], continuity_rows)}

`dormant time`表示 service 窗口内没有任何该请求 op 运行的时间；`service stretch = service time / active wall time`。这两个指标直接反映请求开始后是否被长时间搁置。

![Request continuity](figures/request_continuity_comparison.png)

## 调度与队列指标

{markdown_table(['策略', 'ready queue avg / peak', 'dependency stall', '调度开销', 'critical path', 'DAG parallelism', '跨 device 依赖', '并行利用率'], scheduling_rows)}

## 相对 FCFS 的变化

{markdown_table(relative_headers, relative_rows)}

百分比为 `(策略值 / FCFS - 1) x 100%`。时间、队列、stall 和开销越低越好；吞吐量越高越好。

## Device 占用时间线

绿色表示对应 device 正在执行至少一个 DAG op。虚线表示 `1400 / 0.15` 得到的理论泊松到达窗口末端。该图使用结果 JSON 的 op start/end，而不是 `nvidia-smi` SM utilization。

![Device occupancy comparison](figures/device_occupancy_comparison.png)

{markdown_table(device_headers, device_rows)}

![Device busy comparison](figures/device_busy_comparison.png)

## Dataset 平均 Service Time

各数据集 DAG 结构不同，分数据集 service time 用于确认策略是否只对特定任务形态有效。

{markdown_table(dataset_headers, dataset_rows)}

![Dataset service time comparison](figures/dataset_service_time_comparison.png)

## 新策略 GPU 硬件采样

旧三策略没有连续 `nvidia-smi` CSV，因此硬件采样只比较同一次串行运行中的 RH-SAIL 与 DARC。显存为整卡占用，GPU 3 包含预先存在的其他进程显存。

{markdown_table(['策略', '平均 GPU utilization', '平均 P95 utilization', '单卡最高显存', '平均单卡功耗'], hardware_rows)}

![New scheduler GPU hardware](figures/new_scheduler_gpu_hardware.png)

## 结论

- 五种策略均完成 `1400/1400`，没有 OOM、context length 错误或请求失败。
- **系统总完成时间最短的是 {LABELS[makespan_best]}**：`{metrics.loc[makespan_best, 'makespan']:.1f}s`；**总 token 吞吐量最高的是 {LABELS[throughput_best]}**：`{metrics.loc[throughput_best, 'total_token_throughput']:.1f} token/s`。
- **平均请求完成时间最低的是 {LABELS[completion_best]}**：`{metrics.loc[completion_best, 'avg_completion_time_s']:.1f}s`。该指标同时计入首个 op 前等待和请求开始后的 DAG 完成时间。
- RH-SAIL 将 SAILP 的平均 service time 从 `{sailp['service_time_mean']:.1f}s` 降到 `{rhsail['service_time_mean']:.1f}s`，降低 `{abs(percent_change(rhsail['service_time_mean'], sailp['service_time_mean'])):.2f}%`；平均完成时间从 `{sailp['avg_completion_time_s']:.1f}s` 降到 `{rhsail['avg_completion_time_s']:.1f}s`。
- RH-SAIL 相比 FCFS 将平均完成时间从 `{fcfs['avg_completion_time_s']:.1f}s` 降到 `{rhsail['avg_completion_time_s']:.1f}s`，但系统 makespan 变化为 `{percent_change(rhsail['makespan'], fcfs['makespan']):+.2f}%`，调度开销达到 `{rhsail['scheduler_overhead_pct'] * 100:.2f}%`。它改善了请求级公平性和连续性，但 rollout 成本需要继续优化。
- DARC 的系统 makespan 为 `{darc['makespan']:.1f}s`，平均完成时间为 `{darc['avg_completion_time_s']:.1f}s`，平均 service time 为 `{darc['service_time_mean']:.1f}s`。其价值应同时由系统吞吐量、请求完成时间和连续性指标判断，不能只看平均等待。
- 五次运行总 token 数离散度为 `{token_spread:.2f}%`。吞吐量差异主要反映调度和执行顺序，但仍受非确定性生成 token 数影响。

## 数据文件

- 统一指标：`analysis/scheduler_metrics.csv`
- 请求连续性：`analysis/request_continuity_metrics.csv`
- 相对 FCFS：`analysis/relative_vs_fcfs.csv`
- Device：`analysis/device_busy_comparison.csv`、`analysis/device_intervals_merged.csv`
- Dataset：`analysis/dataset_metrics.csv`
- 新策略硬件采样：`analysis/new_scheduler_gpu_hardware.csv`
- 可复现脚本：`analysis/generate_five_scheduler_report.py`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def build_report(
    summaries: dict[str, dict[str, Any]],
    metrics: pd.DataFrame,
    continuity: pd.DataFrame,
    device: pd.DataFrame,
    dataset: pd.DataFrame,
    relative: pd.DataFrame,
    hardware: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    """Write the canonical five-scheduler technical report in UTF-8."""
    old_info = summaries["fcfs"]["run_info"]
    new_info = summaries["rhsail"]["run_info"]

    core_rows: list[list[str]] = []
    timing_rows: list[list[str]] = []
    continuity_rows: list[list[str]] = []
    scheduling_rows: list[list[str]] = []
    for scheduler in SCHEDULERS:
        row = metrics.loc[scheduler]
        cont = continuity.loc[scheduler]
        core_rows.append(
            [
                LABELS[scheduler],
                f"{int(row['completed'])}/{int(row['count'])}",
                f"{row['makespan']:.1f}s ({row['makespan_minutes']:.1f}min)",
                f"{int(row['total_tokens']):,}",
                f"{row['request_throughput']:.4f} req/s",
                f"{row['total_token_throughput']:.1f} token/s",
                f"{row['output_token_throughput']:.1f} token/s",
            ]
        )
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
        continuity_rows.append(
            [
                LABELS[scheduler],
                f"{cont['avg_active_wall_time_s']:.1f}s",
                f"{cont['avg_dormant_time_s']:.1f}s",
                f"{cont['avg_dormant_fraction'] * 100:.1f}%",
                f"{cont['p95_max_inter_op_gap_s']:.1f}s",
                f"{cont['p95_service_stretch']:.2f}x",
            ]
        )
        scheduling_rows.append(
            [
                LABELS[scheduler],
                f"{row['ready_queue_avg']:.1f} / {int(row['ready_queue_peak'])}",
                f"{row['dependency_stall_mean']:.1f}s",
                f"{row['scheduler_overhead_seconds']:.1f}s "
                f"({row['scheduler_overhead_pct'] * 100:.2f}%)",
                f"{row['critical_path_mean']:.1f}s",
                f"{row['dag_parallelism_mean']:.3f}",
                f"{row['cross_device_dependencies_mean']:.2f}",
                f"{row['parallelism_utilization'] * 100:.2f}%",
            ]
        )

    relative_rows: list[list[str]] = []
    relative_names = {
        "system_makespan": "系统总完成时间",
        "total_token_throughput": "总 token 吞吐量",
        "output_token_throughput": "输出 token 吞吐量",
        "average_waiting_time": "平均等待时间",
        "average_service_time": "平均 service time",
        "average_completion_time": "平均完成时间",
        "p95_service_time": "P95 service time",
        "ready_queue_peak": "Ready queue peak",
        "dependency_stall": "Dependency stall",
        "scheduler_overhead": "调度开销",
    }
    lower_is_better = {
        "system_makespan",
        "average_waiting_time",
        "average_service_time",
        "average_completion_time",
        "p95_service_time",
        "ready_queue_peak",
        "dependency_stall",
        "scheduler_overhead",
    }
    for metric_name in relative_names:
        subset = relative[relative["metric"] == metric_name].set_index("scheduler")
        relative_rows.append(
            [
                relative_names[metric_name],
                "越低越好" if metric_name in lower_is_better else "越高越好",
                *[
                    f"{subset.loc[scheduler, 'vs_fcfs_pct']:+.2f}%"
                    for scheduler in SCHEDULERS[1:]
                ],
            ]
        )

    device_rows: list[list[str]] = []
    for gpu in range(3, 8):
        values = []
        for scheduler in SCHEDULERS:
            value = device[
                (device["scheduler"] == scheduler) & (device["physical_gpu"] == gpu)
            ]["running_time_pct"].iloc[0]
            values.append(f"{value:.2f}%")
        device_rows.append([str(gpu), *values])

    dataset_rows: list[list[str]] = []
    for name in DATASET_ORDER:
        values = []
        for scheduler in SCHEDULERS:
            value = dataset[
                (dataset["scheduler"] == scheduler) & (dataset["dataset"] == name)
            ]["avg_service_time_s"].iloc[0]
            values.append(f"{value:.1f}s")
        dataset_rows.append([name, *values])

    hardware_rows: list[list[str]] = []
    for scheduler in ("rhsail", "darc"):
        subset = hardware[hardware["scheduler"] == scheduler]
        hardware_rows.append(
            [
                LABELS[scheduler],
                f"{subset['avg_gpu_util_pct'].mean():.1f}%",
                f"{subset['p95_gpu_util_pct'].mean():.1f}%",
                f"{subset['max_memory_mib'].max() / 1024.0:.2f} GiB",
                f"{subset['avg_power_w'].mean():.1f} W",
            ]
        )

    makespan_best = metrics["makespan"].idxmin()
    throughput_best = metrics["total_token_throughput"].idxmax()
    completion_best = metrics["avg_completion_time_s"].idxmin()
    fcfs = metrics.loc["fcfs"]
    sailp = metrics.loc["sailp"]
    rhsail = metrics.loc["rhsail"]
    darc = metrics.loc["darc"]
    rhsail_cont = continuity.loc["rhsail"]
    sailp_cont = continuity.loc["sailp"]
    darc_cont = continuity.loc["darc"]
    token_spread = (
        (metrics["total_tokens"].max() - metrics["total_tokens"].min())
        / metrics["total_tokens"].mean()
        * 100.0
    )

    report = f"""# Full first200 五种调度策略系统性能报告

## 技术结论

- 五种策略均完成 `1400/1400`，成功率 100%；日志未发现 OOM、context length、KV cache、CUDA 或 traceback 错误。
- **系统吞吐和 makespan 最优的仍是 {LABELS[makespan_best]}**：makespan `{metrics.loc[makespan_best, 'makespan']:.1f}s`，总吞吐量 `{metrics.loc[throughput_best, 'total_token_throughput']:.1f} token/s`。新策略没有超过该系统级基线。
- **RH-SAIL 修复了 SAILP 最明显的请求搁置问题**：ready queue peak 从 `{int(sailp['ready_queue_peak'])}` 降至 `{int(rhsail['ready_queue_peak'])}`，平均 service time 从 `{sailp['service_time_mean']:.1f}s` 降至 `{rhsail['service_time_mean']:.1f}s`，P95 最大算子间空档从 `{sailp_cont['p95_max_inter_op_gap_s']:.1f}s` 降至 `{rhsail_cont['p95_max_inter_op_gap_s']:.1f}s`。
- RH-SAIL 的代价是更高的首次等待和 rollout 开销；其 makespan 为 `{rhsail['makespan']:.1f}s`，相对 FCFS `{percent_change(rhsail['makespan'], fcfs['makespan']):+.2f}%`，调度开销占 `{rhsail['scheduler_overhead_pct'] * 100:.2f}%`。
- **DARC 当前参数表现不理想**：makespan `{darc['makespan']:.1f}s`，平均完成时间 `{darc['avg_completion_time_s']:.1f}s`，ready queue peak `{int(darc['ready_queue_peak'])}`，P95 最大算子间空档 `{darc_cont['p95_max_inter_op_gap_s']:.1f}s`。它缓解了 SAILP 的极端队列爆炸，但 admission/aging/rollout 尚未形成更好的完成时间权衡。

## 核心系统性能

{markdown_table(['策略', '完成', '系统总完成时间', '总 tokens', '请求吞吐量', '总 token 吞吐量', '输出 token 吞吐量'], core_rows)}

系统总完成时间衡量整批任务排空速度，token/s 衡量硬件产出。五次运行的总 token 数离散度为 `{token_spread:.2f}%`，因此吞吐量差异同时包含调度效果和生成 token 的非确定性。

![五种策略核心性能](figures/scheduler_metrics_comparison.png)

## 请求完成时间与连续性

`等待时间`是请求到达至首个 op 开始；`service time`是首个 op 开始至 DAG 完成；`完成时间 = 等待时间 + service time`。

{markdown_table(['策略', '平均等待', '平均 service', 'P50 service', 'P95 service', 'P99 service', '平均完成时间'], timing_rows)}

请求连续性图把 service 窗口拆成实际有 op 运行的 active wall time 与没有该请求 op 运行的 dormant time。它直接检验“请求开始后是否被长期搁置”。

{markdown_table(['策略', '平均 active wall', '平均 dormant', 'Dormant fraction', 'P95 最大算子间空档', 'P95 service stretch'], continuity_rows)}

![请求连续性与调度开销](figures/request_continuity_comparison.png)

## 调度、队列与 DAG 指标

{markdown_table(['策略', 'Ready queue avg / peak', 'Dependency stall', '调度开销', 'Critical path', 'DAG parallelism', '跨 device 依赖', '并行利用率'], scheduling_rows)}

RH-SAIL 的 active-workflow admission 和连续推进保护显著压低了 ready queue；DARC 仍接纳了较多活跃 DAG，导致等待和尾部继续累积。调度开销是 Python 调度器累计耗时占 makespan 的比例，不包含 vLLM 推理时间。

## 相对 FCFS 的变化

{markdown_table(['指标', '优选方向', 'SJF vs FCFS', 'SAILP vs FCFS', 'RH-SAIL vs FCFS', 'DARC vs FCFS'], relative_rows)}

百分比按 `(策略值 / FCFS - 1) × 100%` 计算。时间、队列、stall 和调度开销越低越好；吞吐量越高越好。

## Device 占用与负载

每个策略顶部的深灰刻线表示 1,400 个 query 的真实泊松到达时刻；绿色区间来自详细结果 JSON 中每个 op 的真实 start/end，表示对应 device 至少有一个 DAG op 正在运行。该图不是 `nvidia-smi` 的 SM utilization；虚线是 `1400 / 0.15` 对应的理论泊松到达窗口末端。

![Device 占用时间线](figures/device_occupancy_comparison.png)

{markdown_table(['Physical GPU', *[LABELS[s] for s in SCHEDULERS]], device_rows)}

各策略的 device busy 均较高，主要差异来自请求执行顺序、DAG 内等待和尾部排空，而不是 GPU 长时间整体空闲。

![每张 GPU 的运行时间比例](figures/device_busy_comparison.png)

## 各数据集 Service Time

每个数据集均包含 200 个请求。对数坐标用于同时展示短 DAG 与长 DAG；跨数据集差异反映 DAG 结构、prompt 长度和生成长度的共同影响。

{markdown_table(['Dataset', *[LABELS[s] for s in SCHEDULERS]], dataset_rows)}

![各数据集平均 service time](figures/dataset_service_time_comparison.png)

## 新策略 GPU 硬件采样

旧三策略没有连续 `nvidia-smi` CSV，因此硬件采样仅比较同一次顺序运行中的 RH-SAIL 与 DARC。显存数值是整张卡占用；GPU 3 包含实验开始前已存在的约 1.1 GiB 其他进程显存。

{markdown_table(['策略', '平均 GPU utilization', '平均 P95 utilization', '单卡最高显存', '平均单卡功耗'], hardware_rows)}

![新策略 GPU 硬件采样](figures/new_scheduler_gpu_hardware.png)

## 实验范围与配置

| 项目 | 配置 |
| --- | --- |
| 数据规模 | 7 个 dataset × 200 = 1,400 requests |
| 数据集 | {', '.join(manifest['datasets'])} |
| 抽样与顺序 | 长度过滤后按 seed `{manifest['seed']}` 随机抽样并全局打乱；五种策略使用同一 questions 文件 |
| 长度过滤 | start prompt `< 14000` tokens；start prompt + op max tokens `<= 15500` |
| 模型 | `{new_info['model_path']}` |
| 推理后端 | real vLLM `{new_info['packages']['vllm']}`，CUDA / NVIDIA A800 |
| GPU | physical GPU `3,4,5,6,7`，共 5 张 |
| 上下文与输出 | max model len `{new_info['max_model_len']}`；每个 DAG op 最多 `{summaries['rhsail']['output_max_tokens']}` output tokens |
| GPU memory utilization | `{new_info['gpu_memory_utilization']}` |
| 到达过程 | Poisson `0.15 req/s`；arrival batch size `1`；seed `{summaries['rhsail']['arrival_seed']}` |
| 策略 | FCFS、SJF、SAILP、RH-SAIL、DARC |
| 代码版本 | 旧三策略 `{old_info['git_commit']}`；新两策略 `{new_info['git_commit']}` |

## 方法与复现

- makespan、tokens、等待、service、队列和调度开销来自每次运行的 summary JSON。
- P50/P95/P99、请求连续性、device union busy interval 和分数据集指标由 1,400 条详细请求 JSON 重新计算。
- dormant time 使用请求全部 op 区间的时间并集计算，避免并行 op 被重复计时。
- GPU utilization、显存和功耗按 `runner.log` 中 RH-SAIL/DARC 的起止时间切分 `gpu_metrics.csv`。
- 所有派生 CSV、图表和生成脚本位于本报告相邻的 `analysis/` 与 `figures/`。

## 限制与稳健性

- 每个策略仅运行一次；当前差异是描述性结果，不能视为带置信区间的稳定排序。
- LLM 生成 token 数存在非确定性。五次运行使用相同输入与到达 seed，但总 token 数仍有 `{token_spread:.2f}%` 离散。
- 新旧策略来自不同提交，但使用相同数据、模型、GPU 数、到达过程和内存参数；代码提交差异仍是潜在混杂因素。
- SAIL affinity 目前是软放置信号；没有可验证的跨 worker KV 状态迁移，因此不能把性能变化解释为真实 KV cache reuse 收益。
- `gpu-memory-utilization=0.75` 是 vLLM 的缓存预算参数，并非 `nvidia-smi` 整卡显存硬上限。

## 建议下一步

1. 以 RH-SAIL 为主线，先降低 rollout 调度开销，并扫描 active DAG limit、candidate K 和 rollout horizon。
2. 为 RH-SAIL 增加至少 3 个 arrival seed 的重复实验，报告均值、标准差和 P95 完成时间。
3. DARC 优先收紧 admission，增强已启动 workflow 的 completion commitment；当前版本不建议作为正式性能方案。
4. 在确认 vLLM prefix caching 可观测后，再单独做 SAIL affinity 的开关消融，区分连续性收益与真实 cache reuse 收益。

## 数据文件

- 统一指标：`analysis/scheduler_metrics.csv`
- 请求连续性：`analysis/request_continuity_metrics.csv`
- 相对 FCFS：`analysis/relative_vs_fcfs.csv`
- Device：`analysis/device_busy_comparison.csv`、`analysis/device_intervals_merged.csv`
- Dataset：`analysis/dataset_metrics.csv`
- 新策略硬件采样：`analysis/new_scheduler_gpu_hardware.csv`
- 可复现脚本：`analysis/generate_five_scheduler_report.py`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def write_chart_map() -> None:
    chart_map = """# Chart map

| Report section | Analytical question | Chart family | Fields | Supported claim |
| --- | --- | --- | --- | --- |
| 核心系统性能 | 哪个策略排空最快、吞吐最高？ | Grouped small-multiple bars | makespan, token/s, wait, service, completion | 系统效率与请求时间的权衡 |
| 请求完成时间与连续性 | 请求启动后是否被搁置？ | Grouped small-multiple bars | P95 service, max gap, dormant fraction, overhead | RH-SAIL 显著改善 SAILP 连续性 |
| Device 占用与负载 | GPU 是否长时间空闲？ | Interval timeline | op start/end by device | 差异主要不来自整体 GPU 空闲 |
| Device 占用与负载 | 各卡运行时间是否均衡？ | Grouped bars | device busy ratio | 五种策略均保持较高设备占用 |
| 各数据集 Service Time | 改进是否局限于特定任务？ | Grouped log-scale bars | average service by dataset | 不同 DAG 结构对策略反应不同 |
| 新策略 GPU 硬件采样 | 两个新策略的硬件负载是否异常？ | Grouped bars | utilization, memory, power | 两次运行均保持高负载且无 OOM |
"""
    (ANALYSIS_DIR / "chart_map.md").write_text(chart_map, encoding="utf-8")


def publish_artifacts() -> None:
    PUBLISH_ROOT.mkdir(parents=True, exist_ok=True)
    publish_figures = PUBLISH_ROOT / "figures"
    publish_analysis = PUBLISH_ROOT / "analysis"
    publish_figures.mkdir(parents=True, exist_ok=True)
    publish_analysis.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPORT_PATH, PUBLISH_ROOT / "report.md")
    for path in FIGURE_DIR.glob("*.png"):
        shutil.copy2(path, publish_figures / path.name)
    for name in [
        "scheduler_metrics.csv",
        "request_continuity_metrics.csv",
        "relative_vs_fcfs.csv",
        "device_busy_comparison.csv",
        "device_intervals_merged.csv",
        "dataset_metrics.csv",
        "new_scheduler_gpu_hardware.csv",
        "chart_map.md",
        "generate_five_scheduler_report.py",
        "validate_five_scheduler_report.py",
    ]:
        shutil.copy2(ANALYSIS_DIR / name, publish_analysis / name)


def main() -> None:
    configure_plot_style()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_json(REPORT_ROOT / "manifest_medium_first200.json")
    summaries: dict[str, dict[str, Any]] = {}
    details: dict[str, list[dict[str, Any]]] = {}
    device_intervals: dict[str, dict[str, list[tuple[float, float]]]] = {}
    metrics_rows: list[dict[str, Any]] = []
    continuity_rows: list[dict[str, Any]] = []
    device_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []

    for scheduler in SCHEDULERS:
        summary = load_json(summary_path(scheduler))
        rows = load_json(detail_path(scheduler))
        if (
            len(rows) != EXPECTED_REQUESTS
            or int(summary.get("completed", 0)) != EXPECTED_REQUESTS
            or float(summary.get("success_rate", 0.0)) != 1.0
        ):
            raise RuntimeError(
                f"Incomplete result for {scheduler}: rows={len(rows)}, "
                f"completed={summary.get('completed')}, success={summary.get('success_rate')}"
            )
        if any(request.get("status") != "completed" for request in rows):
            raise RuntimeError(f"Non-completed request found for {scheduler}")

        summaries[scheduler] = summary
        details[scheduler] = rows
        merged_by_device = extract_device_intervals(rows)
        device_intervals[scheduler] = merged_by_device

        service_times = [float(request["service_time"]) for request in rows]
        completion_times = [float(request["latency"]) for request in rows]
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

        continuity_values = [request_continuity(request) for request in rows]
        continuity_rows.append(
            {
                "scheduler": scheduler,
                "avg_active_wall_time_s": float(
                    np.mean([value["active_wall_time_s"] for value in continuity_values])
                ),
                "avg_dormant_time_s": float(
                    np.mean([value["dormant_time_s"] for value in continuity_values])
                ),
                "avg_dormant_fraction": float(
                    np.mean([value["dormant_fraction"] for value in continuity_values])
                ),
                "avg_max_inter_op_gap_s": float(
                    np.mean([value["max_inter_op_gap_s"] for value in continuity_values])
                ),
                "p95_max_inter_op_gap_s": percentile(
                    [value["max_inter_op_gap_s"] for value in continuity_values], 95
                ),
                "avg_service_stretch": float(
                    np.mean([value["service_stretch"] for value in continuity_values])
                ),
                "p95_service_stretch": percentile(
                    [value["service_stretch"] for value in continuity_values], 95
                ),
            }
        )

        for worker in ["0", "1", "2", "3", "4"]:
            spans = merged_by_device.get(worker, [])
            device_rows.append(
                {
                    "scheduler": scheduler,
                    "internal_device": int(worker),
                    "physical_gpu": PHYSICAL_GPUS[worker],
                    "busy_seconds": float(summary["device_busy_seconds"][worker]),
                    "union_busy_seconds": sum(end - start for start, end in spans),
                    "running_time_pct": float(summary["device_busy_pct"][worker]) * 100.0,
                    "output_tokens_per_second": float(
                        summary["device_output_tokens_per_second"][worker]
                    ),
                }
            )
            for start, end in spans:
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
        for name in DATASET_ORDER:
            dataset_requests = grouped[name]
            if len(dataset_requests) != 200:
                raise RuntimeError(f"{scheduler}/{name} has {len(dataset_requests)} requests")
            token_rows = [request_tokens(request) for request in dataset_requests]
            service = [float(request["service_time"]) for request in dataset_requests]
            waiting = [float(request["idle_time"]) for request in dataset_requests]
            completion = [float(request["latency"]) for request in dataset_requests]
            dataset_rows.append(
                {
                    "scheduler": scheduler,
                    "dataset": name,
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

    metrics = pd.DataFrame(metrics_rows).set_index("scheduler")
    continuity = pd.DataFrame(continuity_rows).set_index("scheduler")
    device = pd.DataFrame(device_rows)
    dataset = pd.DataFrame(dataset_rows)
    intervals = pd.DataFrame(interval_rows)
    hardware = load_gpu_hardware_summary()

    relative_specs = [
        ("system_makespan", "makespan"),
        ("total_token_throughput", "total_token_throughput"),
        ("output_token_throughput", "output_token_throughput"),
        ("average_waiting_time", "waiting_time_mean"),
        ("average_service_time", "service_time_mean"),
        ("average_completion_time", "avg_completion_time_s"),
        ("p95_service_time", "p95_service_time_s"),
        ("ready_queue_peak", "ready_queue_peak"),
        ("dependency_stall", "dependency_stall_mean"),
        ("scheduler_overhead", "scheduler_overhead_seconds"),
    ]
    relative_rows = []
    for metric_name, field in relative_specs:
        baseline = float(metrics.loc["fcfs", field])
        for scheduler in SCHEDULERS[1:]:
            relative_rows.append(
                {
                    "metric": metric_name,
                    "scheduler": scheduler,
                    "vs_fcfs_pct": percent_change(float(metrics.loc[scheduler, field]), baseline),
                }
            )
    relative = pd.DataFrame(relative_rows)

    metrics.to_csv(ANALYSIS_DIR / "scheduler_metrics.csv", encoding="utf-8-sig")
    continuity.to_csv(ANALYSIS_DIR / "request_continuity_metrics.csv", encoding="utf-8-sig")
    relative.to_csv(ANALYSIS_DIR / "relative_vs_fcfs.csv", index=False, encoding="utf-8-sig")
    device.to_csv(ANALYSIS_DIR / "device_busy_comparison.csv", index=False, encoding="utf-8-sig")
    intervals.to_csv(
        ANALYSIS_DIR / "device_intervals_merged.csv", index=False, encoding="utf-8-sig"
    )
    dataset.to_csv(ANALYSIS_DIR / "dataset_metrics.csv", index=False, encoding="utf-8-sig")
    hardware.to_csv(
        ANALYSIS_DIR / "new_scheduler_gpu_hardware.csv", index=False, encoding="utf-8-sig"
    )

    plot_core_metrics(metrics)
    plot_continuity(metrics, continuity)
    plot_device_occupancy(summaries, device_intervals, details)
    plot_device_busy(device)
    plot_dataset_service(dataset)
    plot_gpu_hardware(hardware)
    build_report(summaries, metrics, continuity, device, dataset, relative, hardware, manifest)
    write_chart_map()
    publish_artifacts()

    print(f"report={REPORT_PATH}")
    print(f"published_report={PUBLISH_ROOT / 'report.md'}")
    print(f"figures={FIGURE_DIR}")
    print(f"metrics={ANALYSIS_DIR / 'scheduler_metrics.csv'}")


if __name__ == "__main__":
    main()
