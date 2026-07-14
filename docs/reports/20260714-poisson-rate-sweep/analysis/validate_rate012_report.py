#!/usr/bin/env python3
"""Validate the comprehensive 0.12 req/s report and supporting artifacts."""

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


def main() -> None:
    for scheduler in ["fcfs", "sjf", "rhsail"]:
        root = RUN_ROOT / "rate012" / scheduler
        details = list(root.glob("*_run1.json"))
        summaries = list(root.glob("*_run1_summary.json"))
        assert len(details) == 1 and len(summaries) == 1
        rows = json.loads(details[0].read_text(encoding="utf-8"))
        summary = json.loads(summaries[0].read_text(encoding="utf-8"))
        assert len(rows) == 1400
        assert summary["completed"] == 1400 and summary["success_rate"] == 1.0
        assert all(row["status"] == "completed" for row in rows)

    with (PUBLISH_ROOT / "analysis" / "rate_metrics.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        metrics = list(csv.DictReader(file))
    assert len(metrics) == 6
    assert {(row["rate_label"], row["scheduler"]) for row in metrics} == {
        (rate, scheduler)
        for rate in ["0.12", "0.15"]
        for scheduler in ["fcfs", "sjf", "rhsail"]
    }
    for row in metrics:
        for key, value in row.items():
            if key not in {"rate_label", "scheduler"}:
                assert math.isfinite(float(value)), (key, value)

    report = PUBLISH_ROOT / "report.md"
    text = report.read_text(encoding="utf-8")
    assert "Poisson 0.12 req/s 三种调度策略综合报告" in text
    assert "0.12 下没有一个新策略全面优于 FCFS" in text
    assert "ready_queue" in text and "不能解释为系统总 backlog" in text
    for bad in ["锛", "绛", "鏁", "璋冨害绛"]:
        assert bad not in text

    dataset = PUBLISH_ROOT / "analysis" / "rate012_dataset_metrics.csv"
    assert dataset.is_file()
    with dataset.open(encoding="utf-8-sig", newline="") as file:
        dataset_rows = list(csv.DictReader(file))
    assert len(dataset_rows) == 21
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    assert len(links) == 9
    for link in links:
        path = PUBLISH_ROOT / link
        assert path.is_file(), path
        with Image.open(path) as image:
            assert image.width >= 1200 and image.height >= 500
            assert any(low != high for low, high in ImageStat.Stat(image.convert("RGB")).extrema)

    print("validation=passed")
    print("assessment=0.12_high_utilization_not_congested")


if __name__ == "__main__":
    main()
