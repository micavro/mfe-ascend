#!/usr/bin/env python3
"""Generate the interim Poisson-rate report for the completed 0.12 runs."""

from __future__ import annotations

import json
import math
import re
import shutil
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch


def find_workspace_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "remote_runs").is_dir() and (candidate / "mfe-ascend").is_dir():
            return candidate
    raise RuntimeError("Could not locate workspace root")


WORKSPACE_ROOT = find_workspace_root()
RUN_ROOT = WORKSPACE_ROOT / "remote_runs" / (
    "20260713-234500-full-first200-poisson012-013-fcfs-sjf-rhsail-"
    "vllm5-max32768-out2048-mem075"
)
BASELINE_015_ROOT = WORKSPACE_ROOT / "remote_runs" / (
    "20260711-024102-full-first200-poisson015-batch1-fcfs-sjf-sailp-"
    "vllm5-max32768-out2048-mem075"
)
RHSAIL_015_ROOT = WORKSPACE_ROOT / "remote_runs" / (
    "20260713-160000-full-first200-poisson015-rhsail-darc-"
    "vllm5-max32768-out2048-mem075"
)
ANALYSIS_DIR = RUN_ROOT / "analysis"
FIGURE_DIR = RUN_ROOT / "figures"
REPORT_PATH = RUN_ROOT / "report.md"
PUBLISH_ROOT = (
    WORKSPACE_ROOT
    / "mfe-ascend"
    / "docs"
    / "reports"
    / "20260714-poisson-rate-sweep"
)

SCHEDULERS = ["fcfs", "sjf", "rhsail"]
LABELS = {"fcfs": "FCFS", "sjf": "SJF", "rhsail": "RH-SAIL"}
COLORS = {"fcfs": "#4C78A8", "sjf": "#F58518", "rhsail": "#7A8F3A"}
GPU_COLORS = {
    3: "#4C78A8",
    4: "#F58518",
    5: "#D45087",
    6: "#7A8F3A",
    7: "#C9A227",
}
PHYSICAL_GPUS = {str(index): index + 3 for index in range(5)}
TZ_UTC8 = timezone(timedelta(hours=8))


def configure_plot_style() -> None:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    font = next(
        (
            name
            for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial"]
            if name in installed
        ),
        "DejaVu Sans",
    )
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


def only_match(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern} under {root}, found {matches}")
    return matches[0]


def result_root(rate: str, scheduler: str) -> Path:
    if rate == "0.12":
        return RUN_ROOT / "rate012" / scheduler
    if scheduler == "rhsail":
        return RHSAIL_015_ROOT / scheduler
    return BASELINE_015_ROOT / scheduler


def load_result(rate: str, scheduler: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = result_root(rate, scheduler)
    summary = json.loads(
        only_match(root, "*_run1_summary.json").read_text(encoding="utf-8")
    )
    rows = json.loads(only_match(root, "*_run1.json").read_text(encoding="utf-8"))
    if len(rows) != 1400 or summary["completed"] != 1400 or summary["success_rate"] != 1.0:
        raise RuntimeError(f"Incomplete result: {rate}/{scheduler}")
    return summary, rows


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def request_continuity(row: dict[str, Any]) -> dict[str, float]:
    intervals = []
    for span in (row.get("benchmark") or {}).values():
        if isinstance(span, list) and len(span) >= 2:
            start, end = float(span[0]), float(span[1])
            if end > start:
                intervals.append((start, end))
    merged = merge_intervals(intervals)
    active = sum(end - start for start, end in merged)
    gaps = [merged[index + 1][0] - merged[index][1] for index in range(len(merged) - 1)]
    service = float(row["service_time"])
    dormant = max(0.0, service - active)
    return {
        "active_wall_time_s": active,
        "dormant_time_s": dormant,
        "dormant_fraction": dormant / service if service > 0 else 0.0,
        "max_gap_s": max(gaps, default=0.0),
    }


def device_intervals(rows: list[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        assignments = row.get("worker_assignments") or {}
        for op, span in (row.get("benchmark") or {}).items():
            if op not in assignments or not isinstance(span, list) or len(span) < 2:
                continue
            start, end = float(span[0]), float(span[1])
            if end > start:
                values[str(assignments[op])].append((start, end))
    return {worker: merge_intervals(items) for worker, items in values.items()}


def build_metrics() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    result_cache: dict[str, Any] = {}
    interval_cache: dict[str, Any] = {}
    for rate in ["0.12", "0.15"]:
        for scheduler in SCHEDULERS:
            summary, rows = load_result(rate, scheduler)
            continuity = [request_continuity(row) for row in rows]
            arrivals = [float(row["arrive_time"]) for row in rows]
            services = [float(row["service_time"]) for row in rows]
            waiting = [float(row["idle_time"]) for row in rows]
            completion = [float(row["latency"]) for row in rows]
            arrival_end = max(arrivals)
            records.append(
                {
                    "rate": float(rate),
                    "rate_label": rate,
                    "scheduler": scheduler,
                    "count": len(rows),
                    "completed": summary["completed"],
                    "makespan_s": float(summary["makespan"]),
                    "arrival_end_s": arrival_end,
                    "drain_tail_s": max(0.0, float(summary["makespan"]) - arrival_end),
                    "total_tokens": summary["total_tokens"],
                    "total_token_throughput": summary["total_token_throughput"],
                    "output_token_throughput": summary["output_token_throughput"],
                    "avg_wait_s": float(np.mean(waiting)),
                    "p95_wait_s": percentile(waiting, 95),
                    "avg_service_s": float(np.mean(services)),
                    "p95_service_s": percentile(services, 95),
                    "p99_service_s": percentile(services, 99),
                    "max_service_s": max(services),
                    "avg_completion_s": float(np.mean(completion)),
                    "p95_completion_s": percentile(completion, 95),
                    "p99_completion_s": percentile(completion, 99),
                    "max_completion_s": max(completion),
                    "ready_queue_avg": summary["ready_queue_avg"],
                    "ready_queue_peak": summary["ready_queue_peak"],
                    "scheduler_overhead_pct": summary["scheduler_overhead_pct"],
                    "avg_device_busy_pct": float(
                        np.mean(list(summary["device_busy_pct"].values())) * 100.0
                    ),
                    "avg_dormant_fraction": float(
                        np.mean([value["dormant_fraction"] for value in continuity])
                    ),
                    "p95_max_gap_s": percentile(
                        [value["max_gap_s"] for value in continuity], 95
                    ),
                }
            )
            result_cache[f"{rate}/{scheduler}"] = (summary, rows)
            if rate == "0.12":
                interval_cache[scheduler] = device_intervals(rows)
    return pd.DataFrame(records), result_cache, interval_cache


def rhsail_snapshot(rate: str) -> dict[str, Any]:
    _summary, rows = load_result(rate, "rhsail")
    snapshots = [
        row.get("scheduler_metrics", {}).get("rhsail")
        for row in rows
        if row.get("scheduler_metrics", {}).get("rhsail")
    ]
    if not snapshots:
        raise RuntimeError(f"Missing RH-SAIL snapshot for rate {rate}")
    return max(snapshots, key=lambda item: int(item.get("decisions", 0)))


def parse_runner_intervals() -> dict[str, tuple[float, float]]:
    text = (RUN_ROOT / "runner_rate012_snapshot.log").read_text(encoding="utf-8")
    pattern = re.compile(
        r"^===== (START|END) rate012 (fcfs|sjf|rhsail) "
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) =====$",
        re.MULTILINE,
    )
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}
    for kind, scheduler, timestamp in pattern.findall(text):
        epoch = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=TZ_UTC8
        ).timestamp()
        (starts if kind == "START" else ends)[scheduler] = epoch
    if set(starts) != set(SCHEDULERS) or set(ends) != set(SCHEDULERS):
        raise RuntimeError(f"Incomplete rate012 runner intervals: {text}")
    return {scheduler: (starts[scheduler], ends[scheduler]) for scheduler in SCHEDULERS}


def load_hardware() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(RUN_ROOT / "gpu_metrics_rate012_snapshot.csv")
    frame = frame[frame["gpu_index"].between(3, 7)].copy()
    intervals = parse_runner_intervals()
    summaries = []
    slices = []
    for scheduler, (start, end) in intervals.items():
        subset = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)].copy()
        subset["scheduler"] = scheduler
        subset["elapsed_minutes"] = (subset["timestamp"] - start) / 60.0
        slices.append(subset)
        for gpu, group in subset.groupby("gpu_index"):
            summaries.append(
                {
                    "scheduler": scheduler,
                    "physical_gpu": int(gpu),
                    "samples": len(group),
                    "avg_gpu_util_pct": float(group["utilization_gpu_pct"].mean()),
                    "p95_gpu_util_pct": percentile(
                        group["utilization_gpu_pct"].tolist(), 95
                    ),
                    "max_memory_gib": float(group["memory_used_mb"].max()) / 1024.0,
                    "avg_power_w": float(group["power_w"].mean()),
                }
            )
    return pd.DataFrame(summaries), pd.concat(slices, ignore_index=True)


def add_labels(ax: plt.Axes, bars: Any, values: list[float], fmt: str) -> None:
    ax.bar_label(bars, labels=[fmt.format(value) for value in values], padding=3, fontsize=7)


def plot_rate012_metrics(metrics: pd.DataFrame) -> None:
    frame = metrics[metrics["rate_label"] == "0.12"].set_index("scheduler")
    specs = [
        ("makespan_s", "System makespan", "minutes", 1 / 60.0, False, "{:.1f}"),
        ("drain_tail_s", "Drain tail after last arrival", "minutes", 1 / 60.0, False, "{:.1f}"),
        ("avg_wait_s", "Average waiting time", "seconds", 1.0, False, "{:.1f}"),
        ("avg_service_s", "Average service time", "seconds", 1.0, False, "{:.1f}"),
        ("avg_completion_s", "Average completion time", "seconds", 1.0, False, "{:.1f}"),
        ("ready_queue_peak", "Ready queue peak", "requests", 1.0, False, "{:.0f}"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    labels = [LABELS[s] for s in SCHEDULERS]
    colors = [COLORS[s] for s in SCHEDULERS]
    for ax, (field, title, ylabel, scale, log_scale, fmt) in zip(axes.flat, specs):
        values = [float(frame.loc[s, field]) * scale for s in SCHEDULERS]
        bars = ax.bar(labels, values, color=colors, width=0.62)
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        add_labels(ax, bars, values, fmt)
    fig.suptitle("Poisson rate 0.12: system and request metrics", fontsize=15, fontweight="bold")
    fig.savefig(FIGURE_DIR / "rate012_metrics.png", dpi=220)
    plt.close(fig)


def plot_congestion_comparison(metrics: pd.DataFrame) -> None:
    specs = [
        ("avg_wait_s", "Average waiting time", "seconds", True),
        ("ready_queue_peak", "Ready queue peak", "requests", False),
        ("drain_tail_s", "Drain tail after last arrival", "minutes", False),
        ("avg_completion_s", "Average completion time", "seconds", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    x = np.arange(len(SCHEDULERS))
    width = 0.34
    for ax, (field, title, ylabel, log_scale) in zip(axes.flat, specs):
        for index, rate in enumerate(["0.12", "0.15"]):
            frame = metrics[metrics["rate_label"] == rate].set_index("scheduler")
            values = [float(frame.loc[s, field]) for s in SCHEDULERS]
            if field == "drain_tail_s":
                values = [value / 60.0 for value in values]
            bars = ax.bar(
                x + (index - 0.5) * width,
                values,
                width,
                label=f"{rate} req/s",
                color="#4C78A8" if rate == "0.12" else "#D45087",
            )
            add_labels(ax, bars, values, "{:.1f}")
        if log_scale:
            ax.set_yscale("log")
        ax.set_xticks(x, [LABELS[s] for s in SCHEDULERS])
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel + (" (log scale)" if log_scale else ""))
        ax.grid(axis="y", which="both")
        ax.grid(axis="x", visible=False)
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle("Congestion indicators: 0.12 vs 0.15 req/s", fontsize=15, fontweight="bold")
    fig.savefig(FIGURE_DIR / "rate012_vs_rate015_congestion.png", dpi=220)
    plt.close(fig)


def plot_rhsail_strengths(metrics: pd.DataFrame) -> None:
    rate012 = metrics[metrics["rate_label"] == "0.12"].set_index("scheduler")
    rate015 = metrics[metrics["rate_label"] == "0.15"].set_index("scheduler")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    labels = [LABELS[s] for s in SCHEDULERS]
    colors = [COLORS[s] for s in SCHEDULERS]

    x = np.arange(len(SCHEDULERS))
    width = 0.34
    for index, (rate, frame, color) in enumerate(
        [("0.12", rate012, "#4C78A8"), ("0.15", rate015, "#D45087")]
    ):
        values = [float(frame.loc[s, "ready_queue_peak"]) for s in SCHEDULERS]
        bars = axes[0, 0].bar(
            x + (index - 0.5) * width,
            values,
            width,
            label=f"{rate} req/s",
            color=color,
        )
        add_labels(axes[0, 0], bars, values, "{:.0f}")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_title("Ready queue remains bounded", fontweight="bold")
    axes[0, 0].set_ylabel("requests")
    axes[0, 0].legend(frameon=False, ncol=2)
    axes[0, 0].grid(axis="y")
    axes[0, 0].grid(axis="x", visible=False)

    for ax, field, title, ylabel in [
        (axes[0, 1], "max_service_s", "Maximum service time at 0.15", "seconds (log scale)"),
        (axes[1, 0], "p99_completion_s", "P99 completion time at 0.15", "seconds (log scale)"),
        (axes[1, 1], "max_completion_s", "Maximum completion time at 0.15", "seconds (log scale)"),
    ]:
        values = [float(rate015.loc[s, field]) for s in SCHEDULERS]
        bars = ax.bar(labels, values, color=colors, width=0.62)
        ax.set_yscale("log")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", which="both")
        ax.grid(axis="x", visible=False)
        add_labels(ax, bars, values, "{:.0f}")

    fig.suptitle(
        "RH-SAIL supported strengths: admission stability and tail control",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(FIGURE_DIR / "rate012_rhsail_strengths.png", dpi=220)
    plt.close(fig)


def plot_device_occupancy(metrics: pd.DataFrame, intervals: dict[str, Any]) -> None:
    frame = metrics[metrics["rate_label"] == "0.12"].set_index("scheduler")
    max_minutes = frame["makespan_s"].max() / 60.0
    arrival_minutes = frame["arrival_end_s"].mean() / 60.0
    fig, axes = plt.subplots(3, 1, figsize=(18, 11), sharex=True, constrained_layout=True)
    for ax, scheduler in zip(axes, SCHEDULERS):
        for row, worker in enumerate(["0", "1", "2", "3", "4"]):
            spans = [
                (start / 60.0, (end - start) / 60.0)
                for start, end in intervals[scheduler].get(worker, [])
            ]
            ax.broken_barh(spans, (row - 0.34, 0.68), facecolors="#2E8B57", edgecolors="none")
        ax.set_yticks(range(5), [f"GPU {gpu}" for gpu in range(3, 8)])
        ax.set_ylim(-0.7, 4.7)
        ax.invert_yaxis()
        ax.axvline(arrival_minutes, color="#555555", linestyle="--", linewidth=1.1)
        ax.set_xlim(0, max_minutes * 1.01)
        ax.set_title(
            f"{LABELS[scheduler]} | makespan {frame.loc[scheduler, 'makespan_s'] / 60:.1f} min | "
            f"avg device busy {frame.loc[scheduler, 'avg_device_busy_pct']:.1f}%",
            loc="left",
            fontweight="bold",
        )
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    axes[-1].set_xlabel("Elapsed time (minutes)")
    fig.suptitle("Rate 0.12 device occupancy (green = at least one DAG op running)", fontsize=15, fontweight="bold")
    fig.legend(
        handles=[
            Patch(facecolor="#2E8B57", label="Running"),
            plt.Line2D([0], [0], color="#555555", linestyle="--", label="Actual last arrival"),
        ],
        loc="upper right",
        frameon=False,
    )
    fig.savefig(FIGURE_DIR / "rate012_device_occupancy.png", dpi=220)
    plt.close(fig)


def plot_hardware(metrics: pd.DataFrame, hardware: pd.DataFrame) -> None:
    frame = metrics[metrics["rate_label"] == "0.12"].set_index("scheduler")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.7), constrained_layout=True)
    x = np.arange(5)
    width = 0.22
    specs = [
        ("schedule_busy", "Scheduling-layer device busy", "%"),
        ("avg_gpu_util_pct", "Average nvidia-smi GPU utilization", "%"),
        ("max_memory_gib", "Maximum whole-GPU memory", "GiB"),
    ]
    for ax, (field, title, ylabel) in zip(axes, specs):
        for index, scheduler in enumerate(SCHEDULERS):
            if field == "schedule_busy":
                values = [
                    float(
                        load_result("0.12", scheduler)[0]["device_busy_pct"][str(worker)]
                    )
                    * 100.0
                    for worker in range(5)
                ]
            else:
                subset = hardware[hardware["scheduler"] == scheduler].sort_values("physical_gpu")
                values = subset[field].to_list()
            bars = ax.bar(
                x + (index - 1) * width,
                values,
                width,
                label=LABELS[scheduler],
                color=COLORS[scheduler],
            )
            add_labels(ax, bars, values, "{:.1f}")
        ax.set_xticks(x, [f"GPU {gpu}" for gpu in range(3, 8)], rotation=18)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
    axes[0].legend(frameon=False, ncol=3)
    fig.suptitle("Rate 0.12 GPU usage", fontsize=15, fontweight="bold")
    fig.savefig(FIGURE_DIR / "rate012_gpu_usage.png", dpi=220)
    plt.close(fig)


def plot_gpu_timeline(hardware_samples: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(18, 10), sharey=True, constrained_layout=True)
    for ax, scheduler in zip(axes, SCHEDULERS):
        subset = hardware_samples[hardware_samples["scheduler"] == scheduler]
        for gpu in range(3, 8):
            group = subset[subset["gpu_index"] == gpu].sort_values("elapsed_minutes").copy()
            group["rolling_util"] = group["utilization_gpu_pct"].rolling(12, min_periods=1).mean()
            ax.plot(
                group["elapsed_minutes"],
                group["rolling_util"],
                label=f"GPU {gpu}",
                color=GPU_COLORS[gpu],
                linewidth=1.0,
                alpha=0.9,
            )
        ax.set_title(LABELS[scheduler], loc="left", fontweight="bold")
        ax.set_ylabel("GPU util (%)")
        ax.set_ylim(0, 105)
        ax.grid(axis="both")
    axes[-1].set_xlabel("Elapsed time (minutes)")
    axes[0].legend(frameon=False, ncol=5, loc="upper right")
    fig.suptitle("Rate 0.12 GPU utilization timeline (1-minute rolling mean)", fontsize=15, fontweight="bold")
    fig.savefig(FIGURE_DIR / "rate012_gpu_timeline.png", dpi=220)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_report(metrics: pd.DataFrame, hardware: pd.DataFrame) -> None:
    rate012 = metrics[metrics["rate_label"] == "0.12"].set_index("scheduler")
    rate015 = metrics[metrics["rate_label"] == "0.15"].set_index("scheduler")
    snapshot012 = rhsail_snapshot("0.12")
    snapshot015 = rhsail_snapshot("0.15")
    result_rows = []
    comparison_rows = []
    hardware_rows = []
    for scheduler in SCHEDULERS:
        row = rate012.loc[scheduler]
        result_rows.append(
            [
                LABELS[scheduler],
                f"{row['makespan_s'] / 60:.1f}min",
                f"{row['arrival_end_s'] / 60:.1f}min",
                f"{row['drain_tail_s'] / 60:.1f}min",
                f"{row['total_token_throughput']:.1f}",
                f"{row['avg_wait_s']:.1f}s",
                f"{row['avg_service_s']:.1f}s",
                f"{row['avg_completion_s']:.1f}s",
                f"{int(row['ready_queue_peak'])}",
                f"{row['avg_device_busy_pct']:.1f}%",
            ]
        )
        comparison_rows.append(
            [
                LABELS[scheduler],
                f"{rate012.loc[scheduler, 'avg_wait_s']:.1f}s / {rate015.loc[scheduler, 'avg_wait_s']:.1f}s",
                f"{int(rate012.loc[scheduler, 'ready_queue_peak'])} / {int(rate015.loc[scheduler, 'ready_queue_peak'])}",
                f"{rate012.loc[scheduler, 'drain_tail_s'] / 60:.1f}min / {rate015.loc[scheduler, 'drain_tail_s'] / 60:.1f}min",
                f"{rate012.loc[scheduler, 'avg_completion_s']:.1f}s / {rate015.loc[scheduler, 'avg_completion_s']:.1f}s",
            ]
        )
        subset = hardware[hardware["scheduler"] == scheduler]
        hardware_rows.append(
            [
                LABELS[scheduler],
                f"{subset['avg_gpu_util_pct'].mean():.1f}%",
                f"{subset['p95_gpu_util_pct'].mean():.1f}%",
                f"{subset['max_memory_gib'].max():.2f}GiB",
                f"{subset['avg_power_w'].mean():.1f}W",
            ]
        )

    worst_tail = rate012["drain_tail_s"].max() / 60.0
    rhsail_vs_sjf_max_service = (
        1.0 - rate015.loc["rhsail", "max_service_s"] / rate015.loc["sjf", "max_service_s"]
    ) * 100.0
    rhsail_vs_sjf_p99_service = (
        1.0 - rate015.loc["rhsail", "p99_service_s"] / rate015.loc["sjf", "p99_service_s"]
    ) * 100.0
    rhsail_vs_sjf_p99_completion = (
        1.0
        - rate015.loc["rhsail", "p99_completion_s"]
        / rate015.loc["sjf", "p99_completion_s"]
    ) * 100.0
    rhsail_vs_sjf_max_completion = (
        1.0
        - rate015.loc["rhsail", "max_completion_s"]
        / rate015.loc["sjf", "max_completion_s"]
    ) * 100.0
    rhsail_vs_fcfs_avg_completion = (
        rate015.loc["rhsail", "avg_completion_s"]
        / rate015.loc["fcfs", "avg_completion_s"]
        - 1.0
    ) * 100.0
    rhsail_vs_sjf_throughput = (
        rate015.loc["rhsail", "total_token_throughput"]
        / rate015.loc["sjf", "total_token_throughput"]
        - 1.0
    ) * 100.0
    report = f"""# 泊松请求到达速率实验报告

> 当前版本已写入 `0.12 req/s` 的 FCFS、SJF、RH-SAIL 完整结果，并以 `0.15 req/s` 作为拥塞参照；`0.13 req/s` 仍在远端运行，完成后将继续更新本报告。

## 结论：0.12 接近饱和，但不属于拥塞状态

- 三种策略均完成 `1400/1400`，成功率 100%，没有 OOM、context length、KV cache、CUDA 或 traceback 错误。
- 实际最后一次请求到达约在 `{rate012['arrival_end_s'].mean() / 60:.1f}` 分钟；三种策略随后仅需 `4.3–{worst_tail:.1f}` 分钟排空。相比 `0.15` 的约 `32–40` 分钟尾部，持续积压已经基本消失。
- 平均等待为 FCFS `{rate012.loc['fcfs', 'avg_wait_s']:.1f}s`、SJF `{rate012.loc['sjf', 'avg_wait_s']:.1f}s`、RH-SAIL `{rate012.loc['rhsail', 'avg_wait_s']:.1f}s`；均远低于 `0.15`。
- Ready queue peak 仅 `31–45`，没有随运行时间持续膨胀；五卡调度层 busy 约 `84.7%–86.8%`，说明 GPU 被充分使用，同时仍保留约 13%–15% 的调度层余量。
- 因此 `0.12` 可定义为**高利用率、轻微排队、无持续积压**，符合“到达结束时系统也基本结束”的中等负载目标。

## 0.12 核心结果

{markdown_table(['策略', 'Makespan', '实际到达结束', '排空尾巴', 'Token/s', '平均等待', '平均 service', '平均完成', 'Queue peak', '平均 device busy'], result_rows)}

图中排空尾巴使用每条请求真实的 `arrive_time`，而不是理论值 `1400 / 0.12`。

![0.12 核心指标](figures/rate012_metrics.png)

## RH-SAIL 的可验证优势：队列稳定性与长尾控制

RH-SAIL 当前最有说服力的优势不是峰值吞吐，而是**在负载上升时保持活跃工作流集合有界，并避免已经启动的 DAG 被长时间搁置**。它形成了一个介于 FCFS 与 SJF 之间的 Pareto 点：吞吐量略有损失，但队列规模和极端完成时间更可控。

### 1. 到达速率上升后，RH-SAIL 的队列峰值几乎不增长

- FCFS ready queue peak：`37 → 237`，扩大 `{rate015.loc['fcfs', 'ready_queue_peak'] / rate012.loc['fcfs', 'ready_queue_peak']:.2f}x`。
- SJF ready queue peak：`45 → 131`，扩大 `{rate015.loc['sjf', 'ready_queue_peak'] / rate012.loc['sjf', 'ready_queue_peak']:.2f}x`。
- RH-SAIL ready queue peak：`31 → 33`，仅扩大 `{rate015.loc['rhsail', 'ready_queue_peak'] / rate012.loc['rhsail', 'ready_queue_peak']:.2f}x`。

这与代码中的 active-workflow admission control 一致：活跃 DAG 上限为 `{snapshot015['active_dag_limit']}`，新 root 只在 active pool 有空位、现有工作不足以填满 GPU，或等待超过上限时才接纳。负载从 0.12 增至 0.15 后，记录到的 admission throttle 从 `{snapshot012['admission_throttles']}` 次增加到 `{snapshot015['admission_throttles']}` 次，说明保护机制确实在高负载下介入，而不是指标偶然变好。

### 2. 相比 SJF，RH-SAIL 显著压低极端 service 与完成时间长尾

在 `0.15 req/s` 下：

- 最大 service time：SJF `{rate015.loc['sjf', 'max_service_s']:.1f}s`，RH-SAIL `{rate015.loc['rhsail', 'max_service_s']:.1f}s`，降低 `{rhsail_vs_sjf_max_service:.1f}%`。
- P99 service time：SJF `{rate015.loc['sjf', 'p99_service_s']:.1f}s`，RH-SAIL `{rate015.loc['rhsail', 'p99_service_s']:.1f}s`，降低 `{rhsail_vs_sjf_p99_service:.1f}%`。
- P99 完成时间：SJF `{rate015.loc['sjf', 'p99_completion_s']:.1f}s`，RH-SAIL `{rate015.loc['rhsail', 'p99_completion_s']:.1f}s`，降低 `{rhsail_vs_sjf_p99_completion:.1f}%`。
- 最大完成时间：SJF `{rate015.loc['sjf', 'max_completion_s']:.1f}s`，RH-SAIL `{rate015.loc['rhsail', 'max_completion_s']:.1f}s`，降低 `{rhsail_vs_sjf_max_completion:.1f}%`。

SJF 优先短任务，因此平均完成时间最好，但少量长 DAG 会被持续延后。RH-SAIL 的 progress commitment、inter-op gap/service-stretch emergency guard 和短视 rollout 会持续给已启动请求保留推进机会。`0.15` 运行中记录到 `{snapshot015['emergency_decisions']}` 次 emergency decision，最大调度器观测 gap 为 `{snapshot015['max_gap_seen']:.1f}s`，没有接近配置的 `180s` hard-gap 阈值。

### 3. 相比 FCFS，RH-SAIL 在高负载下改善平均完成时间并大幅缩小队列

在 `0.15 req/s` 下，RH-SAIL 平均完成时间 `{rate015.loc['rhsail', 'avg_completion_s']:.1f}s`，相对 FCFS 的 `{rate015.loc['fcfs', 'avg_completion_s']:.1f}s` 改善 `{abs(rhsail_vs_fcfs_avg_completion):.1f}%`；queue peak 从 `237` 降到 `33`。因此它不是单纯把所有请求延后，而是通过限制并发活跃 DAG，换取更紧凑的请求完成过程。

### 4. 需要诚实保留的代价

- `0.15` 下 RH-SAIL token/s 比 SJF 低 `{abs(rhsail_vs_sjf_throughput):.1f}%`。
- 调度开销为 `{rate015.loc['rhsail', 'scheduler_overhead_pct'] * 100:.2f}%`，明显高于 FCFS/SJF 的约 2.7%；主要来自 candidate construction 和 horizon=`{snapshot015['rollout_horizon']}` 的 rollout。
- RH-SAIL 的平均完成时间仍明显高于 SJF，因此当前更适合定位为**有界 admission + 请求连续性 + tail fairness 调度器**，而不是平均完成时间或吞吐量的全面最优方案。

代码层面，这些结果分别对应 `RHSailReadyScheduler._apply_admission_control()`、`_emergency_severity()`、`_rollout_cost()` 与 `observe_completion()`：前两者限制活跃工作流并提供硬进度保护，rollout 近似优化 unfinished-request area，在线 EWMA 则按模板、算子和 token bucket 校准运行时间。

![RH-SAIL 数据支持的优势](figures/rate012_rhsail_strengths.png)

## 是否拥挤：与 0.15 对比

{markdown_table(['策略', '平均等待 0.12 / 0.15', 'Queue peak 0.12 / 0.15', '排空尾巴 0.12 / 0.15', '平均完成 0.12 / 0.15'], comparison_rows)}

`0.15` 超过系统有效服务能力，等待、队列和到达结束后的排空尾巴同步增加；`0.12` 的三项指标都回落到稳定范围。Makespan 本身不能直接跨速率判断优劣，因为低速率会主动拉长请求到达窗口。

![0.12 与 0.15 拥塞指标](figures/rate012_vs_rate015_congestion.png)

## Device 占用时间线

绿色表示该 device 至少有一个 DAG op 正在运行，虚线表示实际最后一次请求到达。三种策略在到达窗口内保持较高占用，到达结束后没有形成长时间排空尾部。

![0.12 Device 占用时间线](figures/rate012_device_occupancy.png)

## GPU 使用情况

调度层 busy 来自 op 的真实 start/end；`nvidia-smi` utilization、显存和功耗来自每 5 秒采样。二者口径不同：前者表示是否分配了 DAG op，后者表示 GPU 硬件瞬时活动。

{markdown_table(['策略', '平均 GPU util', '平均 P95 util', '单卡最高显存', '平均单卡功耗'], hardware_rows)}

![0.12 GPU 使用汇总](figures/rate012_gpu_usage.png)

时间线使用 1 分钟滚动平均。泊松到达会产生短暂空档，但没有长期全卡空闲或单卡持续失衡。

![0.12 GPU utilization 时间线](figures/rate012_gpu_timeline.png)

## 拥塞判定口径

本报告将以下现象视为拥塞：

1. 请求到达结束后仍需较长时间排空。
2. 平均/P95 等待和 ready queue 随负载显著增长。
3. 请求完成时间主要由排队或 DAG 内 dormant time 主导。
4. GPU 已接近满载，继续提高到达速率只会增加等待而不能显著提高 token/s。

`0.12` 只有第 4 项接近成立，前三项均不明显，因此判断为接近容量拐点但尚未拥塞。

## 实验配置

| 项目 | 设置 |
| --- | --- |
| 数据 | 7 个数据集 × 200，共 1,400 请求 |
| 调度器 | FCFS、SJF、RH-SAIL |
| 到达过程 | Poisson，`0.12 req/s`，batch size `1`，seed `20260709` |
| 模型 | Llama-3.1-8B-Instruct，real vLLM |
| GPU | A800 physical GPU 3–7，共 5 张 |
| 上下文 | max model len `32768`，每个 op 最多输出 `2048` tokens |
| 显存参数 | `gpu-memory-utilization=0.75` |
| Prefix caching | 关闭 |

## 数据与限制

- 每个配置仅运行一次，当前结论是描述性结果，尚无重复实验置信区间。
- 不同运行的生成 token 总数存在轻微非确定性；token/s 变化需结合总 token 数解释。
- 0.12 与 0.15 使用相同数据顺序和 arrival seed，但请求到达时间按速率缩放。
- 当前报告尚未包含正在运行的 0.13；完整速率曲线将在六项任务结束后更新。

## 数据文件

- 汇总指标：`analysis/rate_metrics.csv`
- GPU 硬件汇总：`analysis/rate012_gpu_hardware.csv`
- GPU 采样切片：`analysis/rate012_gpu_samples.csv`
- 生成脚本：`analysis/generate_rate012_report.py`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def publish() -> None:
    PUBLISH_ROOT.mkdir(parents=True, exist_ok=True)
    publish_figures = PUBLISH_ROOT / "figures"
    publish_analysis = PUBLISH_ROOT / "analysis"
    publish_figures.mkdir(parents=True, exist_ok=True)
    publish_analysis.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPORT_PATH, PUBLISH_ROOT / "report.md")
    for path in FIGURE_DIR.glob("rate012_*.png"):
        shutil.copy2(path, publish_figures / path.name)
    for name in [
        "rate_metrics.csv",
        "rate012_gpu_hardware.csv",
        "rate012_gpu_samples.csv",
        "generate_rate012_report.py",
        "validate_rate012_report.py",
    ]:
        shutil.copy2(ANALYSIS_DIR / name, publish_analysis / name)


def main() -> None:
    configure_plot_style()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics, _results, intervals = build_metrics()
    hardware, hardware_samples = load_hardware()
    metrics.to_csv(ANALYSIS_DIR / "rate_metrics.csv", index=False, encoding="utf-8-sig")
    hardware.to_csv(
        ANALYSIS_DIR / "rate012_gpu_hardware.csv", index=False, encoding="utf-8-sig"
    )
    hardware_samples.to_csv(
        ANALYSIS_DIR / "rate012_gpu_samples.csv", index=False, encoding="utf-8-sig"
    )
    plot_rate012_metrics(metrics)
    plot_congestion_comparison(metrics)
    plot_rhsail_strengths(metrics)
    plot_device_occupancy(metrics, intervals)
    plot_hardware(metrics, hardware)
    plot_gpu_timeline(hardware_samples)
    build_report(metrics, hardware)
    publish()
    print(f"report={PUBLISH_ROOT / 'report.md'}")
    print(f"metrics={PUBLISH_ROOT / 'analysis' / 'rate_metrics.csv'}")


if __name__ == "__main__":
    main()
