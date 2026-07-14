#!/usr/bin/env python3
"""Generate the complete 0.12/0.13/0.15 Poisson-rate comparison report."""

from __future__ import annotations

import json
import math
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# The report runtime does not need pandas' optional native accelerators. Some
# workstation installations ship old numexpr/bottleneck wheels against NumPy 1.x.
sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

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

RATES = ["0.12", "0.13", "0.15"]
HARDWARE_RATES = ["0.12", "0.13"]
SCHEDULERS = ["fcfs", "sjf", "rhsail"]
LABELS = {"fcfs": "FCFS", "sjf": "SJF", "rhsail": "RH-SAIL"}
COLORS = {"fcfs": "#4C78A8", "sjf": "#F58518", "rhsail": "#2E8B57"}
RATE_COLORS = {"0.12": "#4C78A8", "0.13": "#C9A227", "0.15": "#D45087"}
GPU_COLORS = {
    3: "#4C78A8",
    4: "#F58518",
    5: "#D45087",
    6: "#2E8B57",
    7: "#C9A227",
}
TZ_UTC8 = timezone(timedelta(hours=8))


def configure_plot_style() -> None:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next(
        (
            name
            for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial"]
            if name in installed
        ),
        "DejaVu Sans",
    )
    plt.rcParams.update(
        {
            "font.family": selected,
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
    if rate == "0.13":
        return RUN_ROOT / "rate013" / scheduler
    if rate == "0.15" and scheduler == "rhsail":
        return RHSAIL_015_ROOT / scheduler
    if rate == "0.15":
        return BASELINE_015_ROOT / scheduler
    raise ValueError(f"Unsupported rate: {rate}")


def load_result(rate: str, scheduler: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = result_root(rate, scheduler)
    summary = json.loads(only_match(root, "*_run1_summary.json").read_text(encoding="utf-8"))
    rows = json.loads(only_match(root, "*_run1.json").read_text(encoding="utf-8"))
    if len(rows) != 1400 or summary["completed"] != 1400 or summary["success_rate"] != 1.0:
        raise RuntimeError(f"Incomplete result: {rate}/{scheduler}")
    if any(row.get("status", "completed") != "completed" for row in rows):
        raise RuntimeError(f"Non-completed request: {rate}/{scheduler}")
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
        "service_stretch": service / active if active > 0 else 0.0,
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
    results: dict[str, Any] = {}
    intervals: dict[str, Any] = defaultdict(dict)
    for rate in RATES:
        for scheduler in SCHEDULERS:
            summary, rows = load_result(rate, scheduler)
            continuity = [request_continuity(row) for row in rows]
            arrivals = [float(row["arrive_time"]) for row in rows]
            services = [float(row["service_time"]) for row in rows]
            waiting = [float(row["idle_time"]) for row in rows]
            completion = [float(row["latency"]) for row in rows]
            records.append(
                {
                    "rate": float(rate),
                    "rate_label": rate,
                    "scheduler": scheduler,
                    "count": len(rows),
                    "completed": int(summary["completed"]),
                    "success_rate": float(summary["success_rate"]),
                    "makespan_s": float(summary["makespan"]),
                    "arrival_end_s": max(arrivals),
                    "drain_tail_s": max(0.0, float(summary["makespan"]) - max(arrivals)),
                    "total_tokens": int(summary["total_tokens"]),
                    "total_token_throughput": float(summary["total_token_throughput"]),
                    "output_token_throughput": float(summary["output_token_throughput"]),
                    "request_throughput": float(summary["request_throughput"]),
                    "avg_wait_s": float(np.mean(waiting)),
                    "p95_wait_s": percentile(waiting, 95),
                    "avg_service_s": float(np.mean(services)),
                    "p50_service_s": percentile(services, 50),
                    "p95_service_s": percentile(services, 95),
                    "p99_service_s": percentile(services, 99),
                    "max_service_s": max(services),
                    "avg_completion_s": float(np.mean(completion)),
                    "p95_completion_s": percentile(completion, 95),
                    "p99_completion_s": percentile(completion, 99),
                    "max_completion_s": max(completion),
                    "ready_queue_avg": float(summary["ready_queue_avg"]),
                    "ready_queue_peak": int(summary["ready_queue_peak"]),
                    "scheduler_overhead_s": float(summary["scheduler_overhead_seconds"]),
                    "scheduler_overhead_pct": float(summary["scheduler_overhead_pct"]),
                    "avg_device_busy_pct": float(
                        np.mean(list(summary["device_busy_pct"].values())) * 100.0
                    ),
                    "parallelism_utilization": float(summary["parallelism_utilization"]),
                    "dependency_stall_mean_s": float(summary["dependency_stall_mean"]),
                    "critical_path_mean_s": float(summary["critical_path_mean"]),
                    "dag_parallelism_mean": float(summary["dag_parallelism_mean"]),
                    "cross_device_dependencies_mean": float(
                        summary["cross_device_dependencies_mean"]
                    ),
                    "avg_active_wall_s": float(
                        np.mean([value["active_wall_time_s"] for value in continuity])
                    ),
                    "avg_dormant_s": float(
                        np.mean([value["dormant_time_s"] for value in continuity])
                    ),
                    "avg_dormant_fraction": float(
                        np.mean([value["dormant_fraction"] for value in continuity])
                    ),
                    "p95_max_gap_s": percentile(
                        [value["max_gap_s"] for value in continuity], 95
                    ),
                    "p95_service_stretch": percentile(
                        [value["service_stretch"] for value in continuity], 95
                    ),
                }
            )
            results[f"{rate}/{scheduler}"] = (summary, rows)
            intervals[rate][scheduler] = device_intervals(rows)
    return pd.DataFrame(records), results, dict(intervals)


def build_dataset_metrics(results: dict[str, Any]) -> pd.DataFrame:
    records = []
    for rate in RATES:
        for scheduler in SCHEDULERS:
            _summary, rows = results[f"{rate}/{scheduler}"]
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                groups[str(row["dataset"])].append(row)
            for dataset, items in sorted(groups.items()):
                services = [float(item["service_time"]) for item in items]
                records.append(
                    {
                        "rate": float(rate),
                        "rate_label": rate,
                        "scheduler": scheduler,
                        "dataset": dataset,
                        "count": len(items),
                        "avg_wait_s": float(np.mean([item["idle_time"] for item in items])),
                        "avg_service_s": float(np.mean(services)),
                        "p95_service_s": percentile(services, 95),
                        "max_service_s": max(services),
                        "avg_completion_s": float(np.mean([item["latency"] for item in items])),
                    }
                )
    return pd.DataFrame(records)


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


def parse_runner_intervals() -> dict[tuple[str, str], tuple[float, float]]:
    text = (RUN_ROOT / "runner.log").read_text(encoding="utf-8")
    pattern = re.compile(
        r"^===== (START|END) rate(012|013) (fcfs|sjf|rhsail) "
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) =====$",
        re.MULTILINE,
    )
    starts: dict[tuple[str, str], float] = {}
    ends: dict[tuple[str, str], float] = {}
    for kind, rate_code, scheduler, timestamp in pattern.findall(text):
        rate = "0.12" if rate_code == "012" else "0.13"
        epoch = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=TZ_UTC8
        ).timestamp()
        target = starts if kind == "START" else ends
        target[(rate, scheduler)] = epoch
    expected = {(rate, scheduler) for rate in HARDWARE_RATES for scheduler in SCHEDULERS}
    if set(starts) != expected or set(ends) != expected:
        raise RuntimeError(f"Incomplete runner intervals: {text}")
    return {key: (starts[key], ends[key]) for key in expected}


def load_hardware() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(RUN_ROOT / "gpu_metrics.csv")
    frame = frame[frame["gpu_index"].between(3, 7)].copy()
    intervals = parse_runner_intervals()
    summaries = []
    slices = []
    for (rate, scheduler), (start, end) in sorted(intervals.items()):
        subset = frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)].copy()
        if subset.empty:
            raise RuntimeError(f"Missing GPU samples for {rate}/{scheduler}")
        subset["rate_label"] = rate
        subset["scheduler"] = scheduler
        subset["elapsed_minutes"] = (subset["timestamp"] - start) / 60.0
        slices.append(subset)
        for gpu, group in subset.groupby("gpu_index"):
            summaries.append(
                {
                    "rate": float(rate),
                    "rate_label": rate,
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


def add_point_labels(
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
    fmt: str,
    y_offset: int,
) -> None:
    for x, y in zip(xs, ys):
        ax.annotate(
            fmt.format(y),
            (x, y),
            xytext=(0, y_offset),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )


def plot_rate_sweep(metrics: pd.DataFrame) -> None:
    specs = [
        ("makespan_s", "System makespan", "minutes", 1 / 60.0, False, "{:.1f}"),
        ("drain_tail_s", "Drain after last arrival", "minutes", 1 / 60.0, False, "{:.1f}"),
        ("total_token_throughput", "Total token throughput", "tokens/s", 1.0, False, "{:.0f}"),
        ("avg_wait_s", "Average waiting time", "seconds", 1.0, True, "{:.0f}"),
        ("avg_service_s", "Average service time", "seconds", 1.0, False, "{:.0f}"),
        ("avg_completion_s", "Average completion time", "seconds", 1.0, True, "{:.0f}"),
        ("ready_queue_peak", "Ready queue peak", "admitted ready ops", 1.0, True, "{:.0f}"),
        ("avg_device_busy_pct", "Scheduling-layer device busy", "%", 1.0, False, "{:.1f}"),
    ]
    xs = [float(rate) for rate in RATES]
    fig, axes = plt.subplots(2, 4, figsize=(19, 9.5), constrained_layout=True)
    for ax, (field, title, ylabel, scale, log_scale, fmt) in zip(axes.flat, specs):
        for scheduler in SCHEDULERS:
            frame = metrics[metrics["scheduler"] == scheduler].set_index("rate_label")
            values = [float(frame.loc[rate, field]) * scale for rate in RATES]
            ax.plot(xs, values, color=COLORS[scheduler], marker="o", linewidth=2, label=LABELS[scheduler])
        if log_scale:
            ax.set_yscale("log")
        ax.set_xticks(xs, RATES)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Poisson arrival rate (req/s)")
        ax.set_ylabel(ylabel + (" (log scale)" if log_scale else ""))
        ax.grid(axis="both", which="both")
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper center")
    fig.suptitle("Poisson arrival-rate sweep: FCFS, SJF, and RH-SAIL", fontsize=16, fontweight="bold")
    fig.savefig(FIGURE_DIR / "rate_sweep_core_metrics.png", dpi=220)
    plt.close(fig)


def plot_tail_sweep(metrics: pd.DataFrame) -> None:
    specs = [
        ("p95_wait_s", "P95 waiting time"),
        ("p99_service_s", "P99 service time"),
        ("max_service_s", "Maximum service time"),
        ("p99_completion_s", "P99 completion time"),
        ("max_completion_s", "Maximum completion time"),
        ("p95_max_gap_s", "P95 maximum inter-op gap"),
    ]
    xs = [float(rate) for rate in RATES]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for ax, (field, title) in zip(axes.flat, specs):
        for scheduler in SCHEDULERS:
            frame = metrics[metrics["scheduler"] == scheduler].set_index("rate_label")
            values = [float(frame.loc[rate, field]) for rate in RATES]
            ax.plot(xs, values, color=COLORS[scheduler], marker="o", linewidth=2, label=LABELS[scheduler])
        ax.set_yscale("log")
        ax.set_xticks(xs, RATES)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Poisson arrival rate (req/s)")
        ax.set_ylabel("seconds (log scale)")
        ax.grid(axis="both", which="both")
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper center")
    fig.suptitle("Request-tail behavior across arrival rates", fontsize=16, fontweight="bold")
    fig.savefig(FIGURE_DIR / "rate_sweep_tail_metrics.png", dpi=220)
    plt.close(fig)


def plot_dataset_service(dataset_metrics: pd.DataFrame) -> None:
    datasets = sorted(dataset_metrics["dataset"].unique())
    xs = [float(rate) for rate in RATES]
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    for ax, dataset in zip(axes.flat, datasets):
        subset = dataset_metrics[dataset_metrics["dataset"] == dataset]
        for scheduler in SCHEDULERS:
            frame = subset[subset["scheduler"] == scheduler].set_index("rate_label")
            values = [float(frame.loc[rate, "avg_service_s"]) for rate in RATES]
            ax.plot(xs, values, color=COLORS[scheduler], marker="o", linewidth=2, label=LABELS[scheduler])
        ax.set_yscale("log")
        ax.set_xticks(xs, RATES)
        ax.set_title(dataset, fontweight="bold")
        ax.set_xlabel("arrival rate (req/s)")
        ax.set_ylabel("average service (s, log)")
        ax.grid(axis="both", which="both")
    for ax in axes.flat[len(datasets):]:
        ax.axis("off")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    fig.suptitle("Average service time by dataset and arrival rate", fontsize=16, fontweight="bold")
    fig.savefig(FIGURE_DIR / "rate_sweep_dataset_service.png", dpi=220)
    plt.close(fig)


def plot_device_occupancy(metrics: pd.DataFrame, intervals: dict[str, Any]) -> None:
    max_minutes = max(float(metrics["makespan_s"].max()) / 60.0, 205.0)
    fig, axes = plt.subplots(3, 3, figsize=(19, 14), sharex=True, constrained_layout=True)
    for row_index, rate in enumerate(RATES):
        rate_frame = metrics[metrics["rate_label"] == rate].set_index("scheduler")
        for col_index, scheduler in enumerate(SCHEDULERS):
            ax = axes[row_index, col_index]
            for worker_index, worker in enumerate(["0", "1", "2", "3", "4"]):
                spans = [
                    (start / 60.0, (end - start) / 60.0)
                    for start, end in intervals[rate][scheduler].get(worker, [])
                ]
                ax.broken_barh(spans, (worker_index - 0.34, 0.68), facecolors="#2E8B57", edgecolors="none")
            arrival = float(rate_frame.loc[scheduler, "arrival_end_s"]) / 60.0
            ax.axvline(arrival, color="#B23A48", linestyle="--", linewidth=1.1)
            ax.set_ylim(-0.7, 4.7)
            ax.invert_yaxis()
            ax.set_xlim(0, max_minutes)
            ax.set_title(
                f"{rate} req/s · {LABELS[scheduler]}\n"
                f"makespan {rate_frame.loc[scheduler, 'makespan_s'] / 60:.1f} min · "
                f"drain {rate_frame.loc[scheduler, 'drain_tail_s'] / 60:.1f} min",
                fontweight="bold",
                fontsize=10,
            )
            if col_index == 0:
                ax.set_yticks(range(5), [f"GPU {gpu}" for gpu in range(3, 8)])
            else:
                ax.set_yticks(range(5), [])
            if row_index == 2:
                ax.set_xlabel("Elapsed time (minutes)")
            ax.grid(axis="x")
            ax.grid(axis="y", visible=False)
    fig.suptitle("Device occupancy on a shared 0–205 minute axis", fontsize=16, fontweight="bold")
    fig.legend(
        handles=[
            Patch(facecolor="#2E8B57", label="At least one DAG op running"),
            plt.Line2D([0], [0], color="#B23A48", linestyle="--", label="Actual last arrival"),
        ],
        loc="upper right",
        frameon=False,
    )
    fig.savefig(FIGURE_DIR / "rate_sweep_device_occupancy.png", dpi=220)
    plt.close(fig)


def plot_gpu_summary(hardware: pd.DataFrame) -> None:
    specs = [
        ("avg_gpu_util_pct", "Average nvidia-smi GPU utilization", "%"),
        ("p95_gpu_util_pct", "P95 nvidia-smi GPU utilization", "%"),
        ("max_memory_gib", "Maximum whole-GPU memory", "GiB"),
        ("avg_power_w", "Average GPU power", "W"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    x = np.arange(len(SCHEDULERS))
    width = 0.34
    for ax, (field, title, ylabel) in zip(axes.flat, specs):
        for rate_index, rate in enumerate(HARDWARE_RATES):
            values = []
            for scheduler in SCHEDULERS:
                subset = hardware[(hardware["rate_label"] == rate) & (hardware["scheduler"] == scheduler)]
                values.append(float(subset[field].mean() if field != "max_memory_gib" else subset[field].max()))
            bars = ax.bar(
                x + (rate_index - 0.5) * width,
                values,
                width,
                color=RATE_COLORS[rate],
                label=f"{rate} req/s",
            )
            ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3, fontsize=7)
        ax.set_xticks(x, [LABELS[scheduler] for scheduler in SCHEDULERS])
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle("GPU hardware sampling for the 0.12 and 0.13 runs", fontsize=16, fontweight="bold")
    fig.savefig(FIGURE_DIR / "rate_sweep_gpu_hardware.png", dpi=220)
    plt.close(fig)


def plot_gpu_timeline(samples: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(19, 9), sharey=True, constrained_layout=True)
    for row_index, rate in enumerate(HARDWARE_RATES):
        for col_index, scheduler in enumerate(SCHEDULERS):
            ax = axes[row_index, col_index]
            subset = samples[(samples["rate_label"] == rate) & (samples["scheduler"] == scheduler)]
            for gpu, group in subset.groupby("gpu_index"):
                ordered = group.sort_values("elapsed_minutes").copy()
                ordered["rolling_util"] = ordered["utilization_gpu_pct"].rolling(12, min_periods=1).mean()
                ax.plot(
                    ordered["elapsed_minutes"],
                    ordered["rolling_util"],
                    color=GPU_COLORS[int(gpu)],
                    linewidth=0.9,
                    label=f"GPU {int(gpu)}",
                )
            ax.set_title(f"{rate} req/s · {LABELS[scheduler]}", fontweight="bold")
            ax.set_ylim(0, 105)
            ax.set_xlabel("Elapsed time (minutes)")
            if col_index == 0:
                ax.set_ylabel("GPU utilization (%)")
            ax.grid(axis="both")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    fig.suptitle("GPU utilization timeline (1-minute rolling mean)", fontsize=16, fontweight="bold")
    fig.savefig(FIGURE_DIR / "rate_sweep_gpu_timeline.png", dpi=220)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def pct_reduction(new: float, baseline: float) -> float:
    return (1.0 - new / baseline) * 100.0


def build_report(metrics: pd.DataFrame, dataset_metrics: pd.DataFrame, hardware: pd.DataFrame) -> None:
    indexed = metrics.set_index(["rate_label", "scheduler"])
    core_rows = []
    tail_rows = []
    for rate in RATES:
        for scheduler in SCHEDULERS:
            row = indexed.loc[(rate, scheduler)]
            core_rows.append(
                [
                    rate,
                    LABELS[scheduler],
                    "1400/1400",
                    f"{row['makespan_s'] / 60:.1f}",
                    f"{row['arrival_end_s'] / 60:.1f}",
                    f"{row['drain_tail_s'] / 60:.1f}",
                    f"{row['total_token_throughput']:.1f}",
                    f"{row['avg_wait_s']:.1f}",
                    f"{row['avg_service_s']:.1f}",
                    f"{row['avg_completion_s']:.1f}",
                    f"{row['ready_queue_avg']:.1f} / {int(row['ready_queue_peak'])}",
                    f"{row['avg_device_busy_pct']:.1f}%",
                ]
            )
            tail_rows.append(
                [
                    rate,
                    LABELS[scheduler],
                    f"{row['p99_service_s']:.1f}",
                    f"{row['max_service_s']:.1f}",
                    f"{row['p99_completion_s']:.1f}",
                    f"{row['max_completion_s']:.1f}",
                    f"{row['p95_max_gap_s']:.1f}",
                    f"{row['avg_dormant_fraction'] * 100:.1f}%",
                    f"{row['scheduler_overhead_pct'] * 100:.2f}%",
                ]
            )

    load_rows = []
    load_labels = {
        "0.12": "高利用率、基本不拥塞",
        "0.13": "容量边界、轻度拥塞",
        "0.15": "明显过载",
    }
    for rate in RATES:
        frame = metrics[metrics["rate_label"] == rate]
        load_rows.append(
            [
                rate,
                f"{frame['arrival_end_s'].mean() / 60:.1f}min",
                f"{frame['drain_tail_s'].min() / 60:.1f}–{frame['drain_tail_s'].max() / 60:.1f}min",
                f"{frame['avg_wait_s'].min():.1f}–{frame['avg_wait_s'].max():.1f}s",
                f"{frame['avg_device_busy_pct'].min():.1f}–{frame['avg_device_busy_pct'].max():.1f}%",
                load_labels[rate],
            ]
        )

    dataset_rows = []
    for dataset in sorted(dataset_metrics["dataset"].unique()):
        row = [dataset]
        for scheduler in SCHEDULERS:
            frame = dataset_metrics[
                (dataset_metrics["dataset"] == dataset)
                & (dataset_metrics["scheduler"] == scheduler)
            ].set_index("rate_label")
            row.append(" / ".join(f"{frame.loc[rate, 'avg_service_s']:.1f}" for rate in RATES))
        dataset_rows.append(row)

    hardware_rows = []
    for rate in HARDWARE_RATES:
        for scheduler in SCHEDULERS:
            subset = hardware[(hardware["rate_label"] == rate) & (hardware["scheduler"] == scheduler)]
            hardware_rows.append(
                [
                    rate,
                    LABELS[scheduler],
                    f"{subset['avg_gpu_util_pct'].mean():.1f}%",
                    f"{subset['p95_gpu_util_pct'].mean():.1f}%",
                    f"{subset['max_memory_gib'].max():.2f}GiB",
                    f"{subset['avg_power_w'].mean():.1f}W",
                ]
            )

    snapshot_rows = []
    snapshots = {rate: rhsail_snapshot(rate) for rate in RATES}
    for rate in RATES:
        snapshot = snapshots[rate]
        snapshot_rows.append(
            [
                rate,
                str(snapshot["active_dag_limit"]),
                str(snapshot["admission_throttles"]),
                str(snapshot["emergency_decisions"]),
                f"{snapshot['max_gap_seen']:.1f}s",
                f"{snapshot['max_stretch_seen']:.2f}x",
                str(snapshot["runtime_model_buckets"]),
            ]
        )

    fcfs012 = indexed.loc[("0.12", "fcfs")]
    fcfs013 = indexed.loc[("0.13", "fcfs")]
    fcfs015 = indexed.loc[("0.15", "fcfs")]
    sjf013 = indexed.loc[("0.13", "sjf")]
    rhsail013 = indexed.loc[("0.13", "rhsail")]
    sjf015 = indexed.loc[("0.15", "sjf")]
    rhsail015 = indexed.loc[("0.15", "rhsail")]

    report = f"""# 泊松请求到达速率实验报告

> 比较 `0.12 / 0.13 / 0.15 req/s` 下 FCFS、SJF、RH-SAIL。每个配置使用相同的 1,400 个请求、顺序和 arrival seed；本报告由请求级 JSON 与 GPU 采样重新计算。

## 技术结论

- 六个新运行均完成 `1400/1400`，成功率 100%；连同 `0.15` 参照共九组数据，未发现 OOM、context length、KV cache、CUDA 或 traceback 运行错误。
- `0.12` 是**高利用率但基本不拥塞**：到达结束后仅需 `4.3–10.5min` 排空。`0.13` 已进入**容量边界上的轻度拥塞**：尾部扩大到 `11.0–17.9min`，但仍远小于 `0.15` 的 `32.5–39.5min`。
- FCFS 是当前环境中最稳健的基线。速率从 `0.12` 增至 `0.15` 时，其平均 service 仅从 `{fcfs012['avg_service_s']:.1f}s` 增至 `{fcfs015['avg_service_s']:.1f}s`，P99 service 也只从 `{fcfs012['p99_service_s']:.1f}s` 增至 `{fcfs015['p99_service_s']:.1f}s`；恶化主要发生在首次等待，而不是请求启动后的连续性。
- SJF 在三档负载下都降低了平均等待或平均完成时间，但少量长 DAG 会被持续越过。`0.13` 的最大 service 已达到 `{sjf013['max_service_s']:.1f}s`，`0.15` 进一步达到 `{sjf015['max_service_s']:.1f}s`。
- RH-SAIL 没有全面超过 FCFS。它的稳定收益是相对 SJF 修复极端长尾：`0.13` 下 P99 service、最大 service 和 P99 completion 分别降低 `{pct_reduction(rhsail013['p99_service_s'], sjf013['p99_service_s']):.1f}%`、`{pct_reduction(rhsail013['max_service_s'], sjf013['max_service_s']):.1f}%`、`{pct_reduction(rhsail013['p99_completion_s'], sjf013['p99_completion_s']):.1f}%`；`0.15` 下对应降低 `{pct_reduction(rhsail015['p99_service_s'], sjf015['p99_service_s']):.1f}%`、`{pct_reduction(rhsail015['max_service_s'], sjf015['max_service_s']):.1f}%`、`{pct_reduction(rhsail015['p99_completion_s'], sjf015['p99_completion_s']):.1f}%`。
- 代价同样明确：RH-SAIL 的调度开销为约 `11%–12%`，明显高于 FCFS/SJF 的约 `2%–4%`；在 `0.12` 和 `0.13` 下，它的平均完成时间、makespan 与 token/s 均没有形成相对 FCFS 的 Pareto 改进。

## 负载分区

{markdown_table(['到达速率', '实际到达窗口', '排空尾部范围', '平均等待范围', '平均 device busy', '判断'], load_rows)}

`0.13` 是三档中最接近“尽量用满 GPU、但到达结束后不留下很长尾部”的档位，不过它已经略高于稳态有效服务能力，不能再称为完全不拥塞。继续提高到 `0.15` 后，token/s 与 device busy 增益很小，但等待和排空尾部成倍增加。

## 核心性能

{markdown_table(['速率', '策略', '完成', 'Makespan (min)', '到达结束 (min)', '排空 (min)', 'Token/s', '平均等待 (s)', '平均 service (s)', '平均完成 (s)', 'Ready avg / peak', 'Device busy'], core_rows)}

Makespan 不能直接跨速率判断容量，因为较低到达速率本身会拉长实验窗口。负载判定主要看真实到达结束后的排空尾部、等待、队列和 GPU 是否已经接近饱和。

![三档速率核心指标](figures/rate_sweep_core_metrics.png)

## 请求连续性与长尾

{markdown_table(['速率', '策略', 'P99 service', 'Max service', 'P99 completion', 'Max completion', 'P95 max gap', 'Dormant fraction', '调度开销'], tail_rows)}

FCFS 的优势来自简单、work-conserving 且保持请求级先来先服务：它几乎没有预测和 rollout 成本，也很少在请求启动后长期搁置该请求。SJF 优化平均值，但会把压力集中到少数长 DAG。RH-SAIL 的 progress commitment、gap/stretch guard 与 rollout 能修复 SJF 的极端 service 长尾，但 admission 等待和候选计算又抬高了平均值。

![三档速率长尾指标](figures/rate_sweep_tail_metrics.png)

## RH-SAIL 的 admission 指标应如何解释

{markdown_table(['速率', 'Active DAG limit', 'Admission throttle', 'Emergency decision', 'Max observed gap', 'Max stretch', 'Runtime buckets'], snapshot_rows)}

RH-SAIL 在三档速率下的 ready peak 都保持在 `30–33`，而 FCFS/SJF 随负载上升明显增大。这说明 **admission 后的活跃候选集合有界**，与 active DAG limit=`15` 的实现一致；但当前 `ready_queue` 是经过 RH-SAIL admission 过滤后的采样，不是系统原始 backlog。因此不能把“30 个 ready 节点”直接解释成系统只积压了 30 个请求，也不能与 FCFS 的 raw ready peak 作完全同义比较。下一轮应同时记录 raw ready、admitted ready、unadmitted roots 和 oldest admission wait。

当前最准确的定位是：RH-SAIL 是**高压下的有界 admission、请求连续性和 tail-fairness 保护器**，不是当前同构 GPU、关闭 prefix caching 条件下的最高吞吐调度器。

## 各数据集 Service Time

下表单元格依次为 `0.12 / 0.13 / 0.15` 的平均 service time，单位为秒；每个数据集每次运行包含 200 个请求。

{markdown_table(['Dataset', 'FCFS', 'SJF', 'RH-SAIL'], dataset_rows)}

SJF 的极端长尾主要集中在 `swebench_verified` 等长工作流；RH-SAIL 能缩小这类请求的最坏 service，但会使若干中等 DAG 的平均 service 高于 FCFS。这也是“修复尾部但没有改善总体均值”的主要来源。

![各数据集平均 service](figures/rate_sweep_dataset_service.png)

## Device 占用

绿色表示该物理 GPU 至少有一个 DAG op 正在运行，红色虚线表示该次运行真实的最后到达时刻。九个面板严格共享 `0–205min` 横轴，因此可以直接观察不同速率的到达窗口与排空尾部。

![三档速率 Device 占用](figures/rate_sweep_device_occupancy.png)

## GPU 硬件采样

完整连续的 `nvidia-smi` 采样覆盖本次 `0.12` 和 `0.13` 六个运行；旧 `0.15` 三策略实验没有同口径完整硬件 CSV，因此不伪造第三档硬件对比。调度层 device busy 与硬件 utilization 含义不同：前者由 op start/end 合并得到，后者是每 5 秒的 GPU 瞬时活动。

{markdown_table(['速率', '策略', '平均 GPU util', '平均 P95 util', '单卡最高显存', '平均单卡功耗'], hardware_rows)}

![GPU 硬件汇总](figures/rate_sweep_gpu_hardware.png)

![GPU utilization 时间线](figures/rate_sweep_gpu_timeline.png)

## 为什么 FCFS 目前很难超过

1. 当前 5 张 GPU 同构，prefix caching 关闭，SAIL 的设备亲和与状态复用信号没有真实 KV reuse 收益。
2. vLLM 已在单个 operator 内部完成推理调度；MFE 没有 RHRS 所依赖的 prefill/decode component-level batching，因此只借用了 rollout，却没有获得其主要 batching 收益。
3. FCFS 始终向空闲 worker 提供最早的 ready op，开销低且不会因 admission 缩小可选工作集；在当前环境下，这种简单策略已经非常接近吞吐友好的工作保持策略。
4. RH-SAIL 每次进行候选构造和三步 rollout，运行时间预测仍存在误差；它主动为连续性和尾部公平付费，但这些收益目前没有转化为更好的整体吞吐。

## 下一步建议

1. 把 RH-SAIL 改成负载自适应模式：轻载使用 FCFS，只有 raw backlog、排空预测或等待持续超阈值时才启用 admission/rollout。
2. 将 candidate K 和 rollout horizon 扫描到更轻量的组合，目标把 scheduler overhead 控制到 `3%–5%`。
3. 增加 raw/admitted/unadmitted 三层队列指标，避免把 filtered ready queue 当作系统总积压。
4. 增加至少 3 个 arrival/generation seed；当前每个配置只有一次运行，1%–3% 的差异不能视为稳定提升。
5. 若要验证 SAIL/RHRS 的论文收益，需要分别开启真实 prefix caching/state reuse，并把执行接口细化到可批处理的推理组件。

## 实验配置与限制

| 项目 | 设置 |
| --- | --- |
| 数据 | 7 个数据集 × 200，共 1,400 请求 |
| 调度器 | FCFS、SJF、RH-SAIL |
| 到达过程 | Poisson，`0.12 / 0.13 / 0.15 req/s`，batch size `1`，seed `20260709` |
| 模型 | Llama-3.1-8B-Instruct，real vLLM |
| GPU | A800 physical GPU 3–7，共 5 张 |
| 上下文 | max model len `32768`，每个 op 最多输出 `2048` tokens |
| 显存参数 | `gpu-memory-utilization=0.75` |
| Prefix caching | 关闭 |

- 三档速率使用相同数据顺序和 arrival seed，但请求到达时间按速率缩放。
- 每个配置只运行一次，结论是描述性结果，尚无重复实验置信区间。
- 生成长度具有非确定性，各运行总 token 数略有差异；token/s 必须结合总 token 数解释。
- service、continuity 和 dataset 指标由请求级 JSON 重算；GPU 指标按 runner 时间窗口切分硬件采样。

## 数据文件

- 汇总指标：`analysis/rate_metrics.csv`
- 分数据集指标：`analysis/rate_dataset_metrics.csv`
- GPU 硬件汇总：`analysis/rate_gpu_hardware.csv`
- GPU 采样切片：`analysis/rate_gpu_samples.csv`
- 生成脚本：`analysis/generate_rate_sweep_report.py`
- 校验脚本：`analysis/validate_rate_sweep_report.py`
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def publish() -> None:
    PUBLISH_ROOT.mkdir(parents=True, exist_ok=True)
    publish_figures = PUBLISH_ROOT / "figures"
    publish_analysis = PUBLISH_ROOT / "analysis"
    publish_figures.mkdir(parents=True, exist_ok=True)
    publish_analysis.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPORT_PATH, PUBLISH_ROOT / "report.md")
    for path in FIGURE_DIR.glob("rate_sweep_*.png"):
        shutil.copy2(path, publish_figures / path.name)
    for name in [
        "rate_metrics.csv",
        "rate_dataset_metrics.csv",
        "rate_gpu_hardware.csv",
        "rate_gpu_samples.csv",
        "generate_rate_sweep_report.py",
        "validate_rate_sweep_report.py",
    ]:
        path = ANALYSIS_DIR / name
        if path.is_file():
            shutil.copy2(path, publish_analysis / name)


def main() -> None:
    configure_plot_style()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics, results, intervals = build_metrics()
    dataset_metrics = build_dataset_metrics(results)
    hardware, hardware_samples = load_hardware()
    metrics.to_csv(ANALYSIS_DIR / "rate_metrics.csv", index=False, encoding="utf-8-sig")
    dataset_metrics.to_csv(
        ANALYSIS_DIR / "rate_dataset_metrics.csv", index=False, encoding="utf-8-sig"
    )
    hardware.to_csv(
        ANALYSIS_DIR / "rate_gpu_hardware.csv", index=False, encoding="utf-8-sig"
    )
    hardware_samples.to_csv(
        ANALYSIS_DIR / "rate_gpu_samples.csv", index=False, encoding="utf-8-sig"
    )
    plot_rate_sweep(metrics)
    plot_tail_sweep(metrics)
    plot_dataset_service(dataset_metrics)
    plot_device_occupancy(metrics, intervals)
    plot_gpu_summary(hardware)
    plot_gpu_timeline(hardware_samples)
    build_report(metrics, dataset_metrics, hardware)
    publish()
    print(f"report={PUBLISH_ROOT / 'report.md'}")
    print(f"metrics={PUBLISH_ROOT / 'analysis' / 'rate_metrics.csv'}")


if __name__ == "__main__":
    main()
