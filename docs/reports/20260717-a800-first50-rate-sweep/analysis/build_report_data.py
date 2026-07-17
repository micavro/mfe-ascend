#!/usr/bin/env python3
"""Build audited tables and figures for the A800 first50 reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RATE_DIRS = {"0.03": "rate003", "0.12": "rate012", "0.15": "rate015"}
SCHEDULERS = ["fcfs", "sjf", "rhsail"]
SCHEDULER_LABELS = {"fcfs": "FCFS", "sjf": "SJF", "rhsail": "RH-SAIL"}
SCHEDULER_COLORS = {"fcfs": "#2563EB", "sjf": "#D4A017", "rhsail": "#C24174"}

SCRIPT_DIR = Path(__file__).resolve().parent
A800_REPORT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = A800_REPORT_ROOT.parents[2]
HARDWARE_REPORT_ROOT = (
    REPO_ROOT / "docs" / "reports" / "20260717-ascend-a800-first50-performance-comparison"
)


# Source: docs/reports/20260716-ascend-poisson-rate-comparison/report.md.
# These rows were manually transcribed and explicitly audited in that report.
NPU_ROWS = [
    {
        "rate": "0.03", "scheduler": "fcfs", "makespan_s": 12902.2,
        "arrival_end_s": 11929.5, "drain_tail_s": 972.8,
        "input_tokens_per_s": 537.1, "output_tokens_per_s": 61.0,
        "total_tokens_per_s": 598.1, "avg_wait_s": 170.3,
        "avg_run_time_s": 164.0, "avg_service_s": 139.2,
        "p99_service_s": 641.9, "max_service_s": 1097.2,
        "avg_completion_s": 309.5, "p99_completion_s": 969.2,
        "max_completion_s": 1460.3, "p95_max_gap_s": 44.2,
        "scheduler_overhead_s": 215.6, "scheduler_overhead_pct": 1.67,
        "ready_queue_avg": 1.3, "ready_queue_peak": 23,
        "device_busy_pct": 89.0,
    },
    {
        "rate": "0.03", "scheduler": "sjf", "makespan_s": 12171.8,
        "arrival_end_s": 11929.5, "drain_tail_s": 242.3,
        "input_tokens_per_s": 552.8, "output_tokens_per_s": 62.1,
        "total_tokens_per_s": 614.9, "avg_wait_s": 39.6,
        "avg_run_time_s": 157.2, "avg_service_s": 197.7,
        "p99_service_s": 1636.0, "max_service_s": 4907.2,
        "avg_completion_s": 237.3, "p99_completion_s": 2505.0,
        "max_completion_s": 5041.5, "p95_max_gap_s": 224.2,
        "scheduler_overhead_s": 114.3, "scheduler_overhead_pct": 0.94,
        "ready_queue_avg": 1.3, "ready_queue_peak": 28,
        "device_busy_pct": 90.4,
    },
    {
        "rate": "0.03", "scheduler": "rhsail", "makespan_s": 12698.6,
        "arrival_end_s": 11929.4, "drain_tail_s": 769.2,
        "input_tokens_per_s": 536.7, "output_tokens_per_s": 62.1,
        "total_tokens_per_s": 598.8, "avg_wait_s": 187.1,
        "avg_run_time_s": 163.3, "avg_service_s": 164.1,
        "p99_service_s": 647.7, "max_service_s": 988.4,
        "avg_completion_s": 351.1, "p99_completion_s": 2214.3,
        "max_completion_s": 3068.0, "p95_max_gap_s": 57.2,
        "scheduler_overhead_s": 219.0, "scheduler_overhead_pct": 1.72,
        "ready_queue_avg": 0.6, "ready_queue_peak": 21,
        "device_busy_pct": 90.0,
    },
    {
        "rate": "0.12", "scheduler": "fcfs", "makespan_s": 11805.4,
        "arrival_end_s": 2982.6, "drain_tail_s": 8822.8,
        "input_tokens_per_s": 578.1, "output_tokens_per_s": 67.5,
        "total_tokens_per_s": 645.6, "avg_wait_s": 4194.6,
        "avg_run_time_s": 162.7, "avg_service_s": 138.6,
        "p99_service_s": 513.0, "max_service_s": 821.0,
        "avg_completion_s": 4333.2, "p99_completion_s": 8659.4,
        "max_completion_s": 8854.0, "p95_max_gap_s": 53.8,
        "scheduler_overhead_s": 56.8, "scheduler_overhead_pct": 0.48,
        "ready_queue_avg": 121.5, "ready_queue_peak": 257,
        "device_busy_pct": 96.5,
    },
    {
        "rate": "0.12", "scheduler": "sjf", "makespan_s": 11571.0,
        "arrival_end_s": 2982.5, "drain_tail_s": 8588.5,
        "input_tokens_per_s": 584.4, "output_tokens_per_s": 65.9,
        "total_tokens_per_s": 650.4, "avg_wait_s": 2548.2,
        "avg_run_time_s": 157.8, "avg_service_s": 224.2,
        "p99_service_s": 2898.7, "max_service_s": 7331.2,
        "avg_completion_s": 2772.4, "p99_completion_s": 9966.8,
        "max_completion_s": 11042.2, "p95_max_gap_s": 132.5,
        "scheduler_overhead_s": 79.7, "scheduler_overhead_pct": 0.69,
        "ready_queue_avg": 91.2, "ready_queue_peak": 219,
        "device_busy_pct": 95.5,
    },
    {
        "rate": "0.12", "scheduler": "rhsail", "makespan_s": 11928.4,
        "arrival_end_s": 2982.6, "drain_tail_s": 8945.7,
        "input_tokens_per_s": 576.2, "output_tokens_per_s": 67.6,
        "total_tokens_per_s": 643.8, "avg_wait_s": 2822.1,
        "avg_run_time_s": 163.7, "avg_service_s": 173.1,
        "p99_service_s": 761.4, "max_service_s": 820.2,
        "avg_completion_s": 2995.2, "p99_completion_s": 9848.3,
        "max_completion_s": 10341.6, "p95_max_gap_s": 57.8,
        "scheduler_overhead_s": 167.2, "scheduler_overhead_pct": 1.40,
        "ready_queue_avg": 4.3, "ready_queue_peak": 21,
        "device_busy_pct": 96.1,
    },
]


def load_a800(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_frames: list[pd.DataFrame] = []
    dataset_frames: list[pd.DataFrame] = []
    for rate, rate_dir in RATE_DIRS.items():
        directory = run_root / rate_dir
        overall = pd.read_csv(directory / "detailed_brief_overall.csv")
        datasets = pd.read_csv(directory / "detailed_brief_by_dataset.csv")
        overall.insert(0, "rate", rate)
        datasets.insert(0, "rate", rate)
        overall_frames.append(overall)
        dataset_frames.append(datasets)
    overall = pd.concat(overall_frames, ignore_index=True)
    datasets = pd.concat(dataset_frames, ignore_index=True)
    return overall, datasets


def build_gpu_metrics(run_root: Path) -> pd.DataFrame:
    samples = pd.read_csv(run_root / "gpu_metrics.csv")
    timeline = pd.read_csv(run_root / "task_timeline.csv")
    rows: list[dict[str, float | int | str]] = []
    for task in timeline.to_dict(orient="records"):
        window = samples[
            (samples["timestamp"] >= float(task["start_epoch"]))
            & (samples["timestamp"] <= float(task["end_epoch"]))
            & (samples["gpu_index"].between(3, 7))
        ]
        if window.empty:
            raise ValueError(f"no GPU samples for {task['rate']}/{task['scheduler']}")
        rows.append(
            {
                "rate": f"{float(task['rate']):.2f}",
                "scheduler": str(task["scheduler"]),
                "sample_count": len(window),
                "avg_gpu_util_pct": window["utilization_gpu_pct"].mean(),
                "p95_gpu_util_pct": window["utilization_gpu_pct"].quantile(0.95),
                "nonzero_sample_pct": 100.0
                * (window["utilization_gpu_pct"] > 0).mean(),
                "avg_power_w_per_gpu": window["power_w"].mean(),
                "peak_memory_mb": window["memory_used_mb"].max(),
            }
        )
    return pd.DataFrame(rows)


def validate(a800: pd.DataFrame, datasets: pd.DataFrame, gpu: pd.DataFrame) -> None:
    if len(a800) != 9 or set(a800["scheduler"]) != set(SCHEDULERS):
        raise ValueError("expected nine A800 rate/scheduler rows")
    if not ((a800["count"] == 350) & (a800["completed"] == 350)).all():
        raise ValueError("an A800 run is not 350/350")
    if not (a800["success_rate"] >= 1.0 - 1e-9).all():
        raise ValueError("an A800 run is below 100% success")
    if len(datasets) != 63 or not (datasets["count"] == 50).all():
        raise ValueError("expected 63 dataset rows with 50 requests each")
    if len(gpu) != 9 or (gpu["sample_count"] <= 0).any():
        raise ValueError("GPU task-window metrics are incomplete")
    token_delta = (
        a800["input_tokens_per_s"]
        + a800["output_tokens_per_s"]
        - a800["total_tokens_per_s"]
    ).abs()
    if (token_delta > 1e-6).any():
        raise ValueError("token throughput identity failed")
    completion_delta = (
        a800["avg_wait_s"] + a800["avg_service_s"] - a800["avg_completion_s"]
    ).abs()
    if (completion_delta > 1e-6).any():
        raise ValueError("completion = wait + service identity failed")


def save_a800_figure(a800: pd.DataFrame) -> None:
    figure_dir = A800_REPORT_ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    rates = list(RATE_DIRS)
    x = np.arange(len(rates))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for index, scheduler in enumerate(SCHEDULERS):
        frame = a800[a800["scheduler"] == scheduler].set_index("rate").loc[rates]
        offset = (index - 1) * width
        axes[0].bar(
            x + offset,
            frame["total_tokens_per_s"],
            width,
            label=SCHEDULER_LABELS[scheduler],
            color=SCHEDULER_COLORS[scheduler],
            edgecolor="#243042",
            linewidth=0.6,
        )
        axes[1].bar(
            x + offset,
            frame["avg_completion_s"],
            width,
            color=SCHEDULER_COLORS[scheduler],
            edgecolor="#243042",
            linewidth=0.6,
        )
    for ax, ylabel in zip(
        axes,
        ["Total throughput (tokens/s)", "Average completion time (s)"],
    ):
        ax.set_xticks(x, rates)
        ax.set_xlabel("Poisson arrival rate (requests/s)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D7DCE3", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("End-to-end token throughput")
    axes[1].set_title("Average request completion time")
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    fig.suptitle("A800 first50 rate sweep performance", fontsize=15, weight="bold")
    fig.savefig(figure_dir / "a800_rate_performance.png", dpi=220, facecolor="white")
    plt.close(fig)


def build_hardware_tables(a800: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = list(NPU_ROWS[0])
    npu = pd.DataFrame(NPU_ROWS, columns=columns)
    matching_a800 = a800[a800["rate"].isin(["0.03", "0.12"])].copy()
    matching_a800.insert(0, "hardware", "A800")
    npu.insert(0, "hardware", "Ascend 910B4")
    common_columns = ["hardware", *columns]
    combined = pd.concat(
        [npu[common_columns], matching_a800[common_columns]], ignore_index=True
    )

    speedup_rows: list[dict[str, float | str]] = []
    for rate in ["0.03", "0.12"]:
        for scheduler in SCHEDULERS:
            npu_row = npu[(npu["rate"] == rate) & (npu["scheduler"] == scheduler)].iloc[0]
            gpu_row = matching_a800[
                (matching_a800["rate"] == rate)
                & (matching_a800["scheduler"] == scheduler)
            ].iloc[0]
            speedup_rows.append(
                {
                    "rate": rate,
                    "scheduler": scheduler,
                    "total_tokens_per_s_speedup": gpu_row["total_tokens_per_s"]
                    / npu_row["total_tokens_per_s"],
                    "output_tokens_per_s_speedup": gpu_row["output_tokens_per_s"]
                    / npu_row["output_tokens_per_s"],
                    "avg_run_time_speedup": npu_row["avg_run_time_s"]
                    / gpu_row["avg_run_time_s"],
                    "makespan_speedup": npu_row["makespan_s"] / gpu_row["makespan_s"],
                    "drain_speedup": npu_row["drain_tail_s"] / gpu_row["drain_tail_s"],
                    "avg_wait_speedup": npu_row["avg_wait_s"] / gpu_row["avg_wait_s"],
                    "avg_service_speedup": npu_row["avg_service_s"]
                    / gpu_row["avg_service_s"],
                    "avg_completion_speedup": npu_row["avg_completion_s"]
                    / gpu_row["avg_completion_s"],
                }
            )
    return combined, pd.DataFrame(speedup_rows)


def save_hardware_figure(combined: pd.DataFrame) -> None:
    figure_dir = HARDWARE_REPORT_ROOT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    frame = combined[combined["rate"] == "0.12"].copy()
    x = np.arange(len(SCHEDULERS))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    hardware_colors = {"Ascend 910B4": "#8A9099", "A800": "#2563EB"}
    for index, hardware in enumerate(["Ascend 910B4", "A800"]):
        values = frame[frame["hardware"] == hardware].set_index("scheduler").loc[SCHEDULERS]
        offset = (index - 0.5) * width
        axes[0].bar(
            x + offset,
            values["total_tokens_per_s"],
            width,
            label=hardware,
            color=hardware_colors[hardware],
            edgecolor="#243042",
            linewidth=0.6,
        )
        axes[1].bar(
            x + offset,
            values["avg_run_time_s"],
            width,
            color=hardware_colors[hardware],
            edgecolor="#243042",
            linewidth=0.6,
        )
    labels = [SCHEDULER_LABELS[scheduler] for scheduler in SCHEDULERS]
    for ax, ylabel in zip(
        axes,
        ["Total throughput (tokens/s)", "Average op run time per request (s)"],
    ):
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#D7DCE3", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("End-to-end throughput")
    axes[1].set_title("Request op execution time (lower is better)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "Ascend 910B4 vs A800 at 0.12 requests/s",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(figure_dir / "npu_a800_rate012_performance.png", dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()

    a800, datasets = load_a800(run_root)
    gpu = build_gpu_metrics(run_root)
    validate(a800, datasets, gpu)

    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    a800.to_csv(SCRIPT_DIR / "a800_metrics.csv", index=False, encoding="utf-8-sig")
    datasets.to_csv(
        SCRIPT_DIR / "a800_dataset_metrics.csv", index=False, encoding="utf-8-sig"
    )
    gpu.to_csv(SCRIPT_DIR / "a800_gpu_metrics.csv", index=False, encoding="utf-8-sig")
    save_a800_figure(a800)

    hardware, speedups = build_hardware_tables(a800)
    hardware_analysis = HARDWARE_REPORT_ROOT / "analysis"
    hardware_analysis.mkdir(parents=True, exist_ok=True)
    hardware.to_csv(
        hardware_analysis / "hardware_metrics.csv", index=False, encoding="utf-8-sig"
    )
    speedups.to_csv(
        hardware_analysis / "a800_speedups.csv", index=False, encoding="utf-8-sig"
    )
    save_hardware_figure(hardware)
    print("validated 9 A800 runs, 63 dataset rows, and 9 GPU windows")


if __name__ == "__main__":
    main()
