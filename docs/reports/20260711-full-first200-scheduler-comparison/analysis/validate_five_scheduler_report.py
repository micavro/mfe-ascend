#!/usr/bin/env python3
"""Validate five-scheduler source data, derived tables, figures, and report links."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageStat


def find_workspace_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "remote_runs").is_dir() and (candidate / "mfe-ascend").is_dir():
            return candidate
    raise RuntimeError("Could not locate workspace root")


WORKSPACE_ROOT = find_workspace_root()
REPORT_ROOT = WORKSPACE_ROOT / "remote_runs" / (
    "20260711-024102-full-first200-poisson015-batch1-fcfs-sjf-sailp-"
    "vllm5-max32768-out2048-mem075"
)
NEW_RUN_ROOT = WORKSPACE_ROOT / "remote_runs" / (
    "20260713-160000-full-first200-poisson015-rhsail-darc-"
    "vllm5-max32768-out2048-mem075"
)
PUBLISH_ROOT = WORKSPACE_ROOT / "mfe-ascend" / "docs" / "reports" / (
    "20260711-full-first200-scheduler-comparison"
)
SCHEDULERS = ["fcfs", "sjf", "sailp", "rhsail", "darc"]
EXPECTED_DATASETS = {
    "gsm8k",
    "strategyqa",
    "mmlu_pro",
    "math",
    "mbpp",
    "hotpotqa",
    "swebench_verified",
}


def only_match(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    assert len(matches) == 1, (root, pattern, matches)
    return matches[0]


def scheduler_root(scheduler: str) -> Path:
    base = NEW_RUN_ROOT if scheduler in {"rhsail", "darc"} else REPORT_ROOT
    return base / scheduler


def validate_source_results() -> None:
    for scheduler in SCHEDULERS:
        root = scheduler_root(scheduler)
        detail = json.loads(only_match(root, "*_run1.json").read_text(encoding="utf-8"))
        summary = json.loads(
            only_match(root, "*_run1_summary.json").read_text(encoding="utf-8")
        )
        assert len(detail) == 1400
        assert summary["count"] == 1400
        assert summary["completed"] == 1400
        assert summary["success_rate"] == 1.0
        assert all(row["status"] == "completed" for row in detail)
        counts = Counter(row["dataset"] for row in detail)
        assert set(counts) == EXPECTED_DATASETS
        assert set(counts.values()) == {200}
        output_tokens = sum(
            int(metric.get("output_tokens") or 0)
            for row in detail
            for metric in (row.get("op_metrics") or {}).values()
        )
        input_tokens = sum(
            int(metric.get("input_tokens") or 0)
            for row in detail
            for metric in (row.get("op_metrics") or {}).values()
        )
        assert output_tokens == summary["output_tokens"]
        assert input_tokens == summary["input_tokens"]
        completion = sum(float(row["latency"]) for row in detail) / len(detail)
        expected = float(summary["waiting_time_mean"]) + float(summary["service_time_mean"])
        assert math.isclose(completion, expected, rel_tol=0.0, abs_tol=1e-6)


def read_csv(name: str) -> list[dict[str, str]]:
    with (PUBLISH_ROOT / "analysis" / name).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def validate_derived_tables() -> None:
    metrics = read_csv("scheduler_metrics.csv")
    assert [row["scheduler"] for row in metrics] == SCHEDULERS
    for row in metrics:
        assert int(float(row["count"])) == 1400
        assert int(float(row["completed"])) == 1400
        assert float(row["success_rate"]) == 1.0
        for key, value in row.items():
            if key != "scheduler":
                assert math.isfinite(float(value)), (row["scheduler"], key, value)

    device = read_csv("device_busy_comparison.csv")
    assert len(device) == 25
    for row in device:
        assert math.isclose(
            float(row["busy_seconds"]),
            float(row["union_busy_seconds"]),
            rel_tol=0.0,
            abs_tol=1e-6,
        )

    dataset = read_csv("dataset_metrics.csv")
    assert len(dataset) == 35
    assert all(int(row["count"]) == 200 for row in dataset)
    hardware = read_csv("new_scheduler_gpu_hardware.csv")
    assert len(hardware) == 10
    assert {row["scheduler"] for row in hardware} == {"rhsail", "darc"}
    assert all(int(row["samples"]) > 1000 for row in hardware)


def validate_report_and_figures() -> None:
    source_report = REPORT_ROOT / "report.md"
    published_report = PUBLISH_ROOT / "report.md"
    assert source_report.read_bytes() == published_report.read_bytes()
    text = published_report.read_text(encoding="utf-8")
    assert "五种调度策略系统性能报告" in text
    assert all(label in text for label in ["FCFS", "SJF", "SAILP", "RH-SAIL", "DARC"])
    for bad in ["锛", "绛", "鏁", "璋冨害绛"]:
        assert bad not in text, bad

    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    assert len(links) == 6
    for link in links:
        path = PUBLISH_ROOT / link
        assert path.is_file(), path
        with Image.open(path) as image:
            width, height = image.size
            assert width >= 1200 and height >= 600, (path, image.size)
            extrema = ImageStat.Stat(image.convert("RGB")).extrema
            assert any(low != high for low, high in extrema), (path, extrema)

    code_links = re.findall(r"`(analysis/[^`]+)`", text)
    for link in code_links:
        for item in [part.strip("、") for part in link.split("、")]:
            assert (PUBLISH_ROOT / item).is_file(), item


def main() -> None:
    validate_source_results()
    validate_derived_tables()
    validate_report_and_figures()
    print("validation=passed")
    print("confidence=ready_to_share_with_stated_single-run_caveats")


if __name__ == "__main__":
    main()
