#!/usr/bin/env python3
"""Validate the complete Poisson-rate report and its published artifacts."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

from PIL import Image, ImageStat


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
PUBLISH_ROOT = (
    WORKSPACE_ROOT
    / "mfe-ascend"
    / "docs"
    / "reports"
    / "20260714-poisson-rate-sweep"
)
BASELINE_015_ROOT = WORKSPACE_ROOT / "remote_runs" / (
    "20260711-024102-full-first200-poisson015-batch1-fcfs-sjf-sailp-"
    "vllm5-max32768-out2048-mem075"
)
RHSAIL_015_ROOT = WORKSPACE_ROOT / "remote_runs" / (
    "20260713-160000-full-first200-poisson015-rhsail-darc-"
    "vllm5-max32768-out2048-mem075"
)

RATES = ["0.12", "0.13", "0.15"]
SCHEDULERS = ["fcfs", "sjf", "rhsail"]


def result_root(rate: str, scheduler: str) -> Path:
    if rate == "0.12":
        return RUN_ROOT / "rate012" / scheduler
    if rate == "0.13":
        return RUN_ROOT / "rate013" / scheduler
    if scheduler == "rhsail":
        return RHSAIL_015_ROOT / scheduler
    return BASELINE_015_ROOT / scheduler


def one(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    assert len(matches) == 1, (root, pattern, matches)
    return matches[0]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    assert (RUN_ROOT / "DONE").is_file()
    assert (RUN_ROOT / "runner.log").is_file()
    assert (RUN_ROOT / "gpu_metrics.csv").stat().st_size > 4_000_000

    error_pattern = re.compile(
        r"out of memory|\bOOM\b|context length|KV cache|Traceback|CUDA error|NCCL error|fatal",
        re.IGNORECASE,
    )
    logs = sorted(RUN_ROOT.glob("rate*.log"))
    assert len(logs) == 6, logs
    for log in logs:
        assert not error_pattern.search(log.read_text(encoding="utf-8", errors="replace")), log

    metrics_path = PUBLISH_ROOT / "analysis" / "rate_metrics.csv"
    metrics = read_csv(metrics_path)
    assert len(metrics) == 9
    assert {(row["rate_label"], row["scheduler"]) for row in metrics} == {
        (rate, scheduler) for rate in RATES for scheduler in SCHEDULERS
    }
    metric_index = {(row["rate_label"], row["scheduler"]): row for row in metrics}
    for rate in RATES:
        for scheduler in SCHEDULERS:
            row = metric_index[(rate, scheduler)]
            assert int(float(row["count"])) == 1400
            assert int(float(row["completed"])) == 1400
            assert float(row["success_rate"]) == 1.0
            for key, value in row.items():
                if key not in {"rate_label", "scheduler"}:
                    assert math.isfinite(float(value)), (rate, scheduler, key, value)

            root = result_root(rate, scheduler)
            summary = json.loads(one(root, "*_run1_summary.json").read_text(encoding="utf-8"))
            requests = json.loads(one(root, "*_run1.json").read_text(encoding="utf-8"))
            assert len(requests) == 1400
            assert summary["completed"] == 1400 and summary["success_rate"] == 1.0
            assert abs(float(row["makespan_s"]) - float(summary["makespan"])) < 1e-6
            assert abs(
                float(row["total_token_throughput"])
                - float(summary["total_token_throughput"])
            ) < 1e-6
            assert int(float(row["ready_queue_peak"])) == int(summary["ready_queue_peak"])
            assert abs(
                float(row["avg_completion_s"])
                - float(row["avg_wait_s"])
                - float(row["avg_service_s"])
            ) < 1e-6

    assert 10 * 60 < float(metric_index[("0.13", "fcfs")]["drain_tail_s"]) < 20 * 60
    assert float(metric_index[("0.15", "fcfs")]["drain_tail_s"]) > 30 * 60
    assert float(metric_index[("0.13", "rhsail")]["max_service_s"]) < float(
        metric_index[("0.13", "sjf")]["max_service_s"]
    )
    assert float(metric_index[("0.15", "rhsail")]["p99_service_s"]) < float(
        metric_index[("0.15", "sjf")]["p99_service_s"]
    )

    dataset_rows = read_csv(PUBLISH_ROOT / "analysis" / "rate_dataset_metrics.csv")
    assert len(dataset_rows) == 63
    assert len({row["dataset"] for row in dataset_rows}) == 7
    assert all(int(float(row["count"])) == 200 for row in dataset_rows)
    assert {(row["rate_label"], row["scheduler"]) for row in dataset_rows} == {
        (rate, scheduler) for rate in RATES for scheduler in SCHEDULERS
    }

    hardware_rows = read_csv(PUBLISH_ROOT / "analysis" / "rate_gpu_hardware.csv")
    assert len(hardware_rows) == 30
    assert {row["rate_label"] for row in hardware_rows} == {"0.12", "0.13"}
    assert {int(float(row["physical_gpu"])) for row in hardware_rows} == set(range(3, 8))
    assert all(int(float(row["samples"])) > 1_500 for row in hardware_rows)

    sample_path = PUBLISH_ROOT / "analysis" / "rate_gpu_samples.csv"
    assert sample_path.stat().st_size > 4_000_000
    sample_rows = read_csv(sample_path)
    assert len(sample_rows) > 60_000
    assert {row["rate_label"] for row in sample_rows} == {"0.12", "0.13"}

    report = PUBLISH_ROOT / "report.md"
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# 泊松请求到达速率实验报告")
    for phrase in [
        "0.13` 已进入**容量边界上的轻度拥塞",
        "FCFS 是当前环境中最稳健的基线",
        "RH-SAIL 没有全面超过 FCFS",
        "admission 后的活跃候选集合有界",
        "为什么 FCFS 目前很难超过",
    ]:
        assert phrase in text, phrase
    for stale in ["0.13 req/s` 仍在远端运行", "当前报告尚未包含正在运行的 0.13"]:
        assert stale not in text
    for bad in ["锟", "涓夌", "缁煎", "鐨勫"]:
        assert bad not in text

    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    assert len(links) == 6, links
    for link in links:
        path = PUBLISH_ROOT / link
        assert path.is_file(), path
        with Image.open(path) as image:
            assert image.width >= 1200 and image.height >= 700, (path, image.size)
            extrema = ImageStat.Stat(image.convert("RGB")).extrema
            assert any(low != high for low, high in extrema), path

    assert (RUN_ROOT / "report.md").read_bytes() == report.read_bytes()
    for name in ["generate_rate_sweep_report.py", "validate_rate_sweep_report.py"]:
        assert (PUBLISH_ROOT / "analysis" / name).is_file()

    print("validation=passed")
    print("assessment=ready_to_share_with_single-run_caveat")
    print("load_classification=0.12_not_congested;0.13_boundary_mild;0.15_overloaded")


if __name__ == "__main__":
    main()
