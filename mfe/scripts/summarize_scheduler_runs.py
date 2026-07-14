#!/usr/bin/env python3
"""Build a compact, self-contained report from scheduler experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


DISPLAY_NAMES = {
    "fcfs": "FCFS",
    "sjf": "SJF",
    "rhsail": "RH-SAIL",
    "sailp": "SAILP",
    "darc": "DARC",
}


@dataclass(frozen=True)
class RunMetrics:
    scheduler: str
    count: int
    completed: int
    success_rate: float
    makespan_s: float
    arrival_end_s: float
    drain_tail_s: float
    total_tokens_per_s: float
    avg_wait_s: float
    avg_service_s: float
    avg_completion_s: float
    ready_queue_avg: float
    ready_queue_peak: int
    device_busy_pct: float


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _single_file(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {label} in {directory}, found {len(matches)}: "
            f"{[path.name for path in matches]}"
        )
    return matches[0]


def _numeric_mean(values: Iterable[Any]) -> float:
    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        raise ValueError("cannot compute a mean from an empty metric")
    return mean(numbers)


def load_run_metrics(
    run_dir: Path,
    *,
    scheduler: str | None = None,
    expected_count: int | None = None,
    require_complete: bool = True,
) -> RunMetrics:
    """Load one scheduler run and calculate the compact report metrics."""
    summary_path = _single_file(run_dir, "*_summary.json", "summary JSON")
    detail_name = summary_path.name.removesuffix("_summary.json") + ".json"
    detail_path = summary_path.with_name(detail_name)
    if not detail_path.is_file():
        raise ValueError(f"detail JSON matching {summary_path.name} is missing: {detail_path}")

    summary = _load_json(summary_path)
    details = _load_json(detail_path)
    if not isinstance(summary, Mapping):
        raise ValueError(f"summary JSON is not an object: {summary_path}")
    if not isinstance(details, list):
        raise ValueError(f"detail JSON is not a list: {detail_path}")

    resolved_scheduler = str(scheduler or summary.get("scheduler") or run_dir.name).lower()
    count = int(summary.get("count") or 0)
    completed = int(summary.get("completed") or 0)
    success_rate = float(summary.get("success_rate") or 0.0)
    if len(details) != count:
        raise ValueError(
            f"{resolved_scheduler}: detail row count {len(details)} does not match summary count {count}"
        )
    if expected_count is not None and count != expected_count:
        raise ValueError(f"{resolved_scheduler}: count {count}, expected {expected_count}")
    if require_complete and (completed != count or success_rate < 1.0 - 1e-9):
        raise ValueError(
            f"{resolved_scheduler}: incomplete run, completed={completed}/{count}, "
            f"success_rate={success_rate:.6f}"
        )

    valid = [
        row
        for row in details
        if isinstance(row, Mapping)
        and row.get("status", "completed") == "completed"
        and row.get("arrive_time") is not None
        and row.get("done_time") is not None
    ]
    if len(valid) != completed:
        raise ValueError(
            f"{resolved_scheduler}: valid detail rows {len(valid)} do not match completed {completed}"
        )
    if not valid:
        raise ValueError(f"{resolved_scheduler}: no completed detail rows")

    arrival_times = [float(row["arrive_time"]) for row in valid]
    done_times = [float(row["done_time"]) for row in valid]
    arrival_end_s = max(arrival_times) - min(arrival_times)
    derived_makespan_s = max(done_times) - min(arrival_times)
    makespan_s = float(summary.get("makespan") or derived_makespan_s)
    tolerance = max(1e-3, derived_makespan_s * 1e-6)
    if abs(makespan_s - derived_makespan_s) > tolerance:
        raise ValueError(
            f"{resolved_scheduler}: summary makespan {makespan_s:.6f} differs from "
            f"detail-derived makespan {derived_makespan_s:.6f}"
        )

    busy = summary.get("device_busy_pct") or {}
    if not isinstance(busy, Mapping) or not busy:
        raise ValueError(f"{resolved_scheduler}: device_busy_pct is missing")

    return RunMetrics(
        scheduler=resolved_scheduler,
        count=count,
        completed=completed,
        success_rate=success_rate,
        makespan_s=makespan_s,
        arrival_end_s=arrival_end_s,
        drain_tail_s=max(0.0, makespan_s - arrival_end_s),
        total_tokens_per_s=float(summary.get("total_token_throughput") or 0.0),
        avg_wait_s=_numeric_mean(row.get("idle_time") for row in valid),
        avg_service_s=_numeric_mean(row.get("service_time") for row in valid),
        avg_completion_s=_numeric_mean(row.get("latency") for row in valid),
        ready_queue_avg=float(summary.get("ready_queue_avg") or 0.0),
        ready_queue_peak=int(summary.get("ready_queue_peak") or 0),
        device_busy_pct=_numeric_mean(busy.values()) * 100.0,
    )


def collect_metrics(
    output_root: Path,
    schedulers: Sequence[str],
    *,
    expected_count: int | None = None,
    require_complete: bool = True,
) -> list[RunMetrics]:
    rows: list[RunMetrics] = []
    for scheduler in schedulers:
        run_dir = output_root / scheduler
        if not run_dir.is_dir():
            raise ValueError(f"scheduler output directory does not exist: {run_dir}")
        rows.append(
            load_run_metrics(
                run_dir,
                scheduler=scheduler,
                expected_count=expected_count,
                require_complete=require_complete,
            )
        )
    return rows


def markdown_table(rows: Sequence[RunMetrics]) -> str:
    lines = [
        "| Strategy | Done | Success | Makespan(s) | Arrival end(s) | Drain(s) | Tokens/s | "
        "Avg wait(s) | Avg service(s) | Avg completion(s) | Ready avg/peak | Device busy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        name = DISPLAY_NAMES.get(row.scheduler, row.scheduler.upper())
        lines.append(
            f"| {name} | {row.completed}/{row.count} | {row.success_rate * 100:.1f}% | "
            f"{row.makespan_s:.1f} | {row.arrival_end_s:.1f} | {row.drain_tail_s:.1f} | "
            f"{row.total_tokens_per_s:.1f} | {row.avg_wait_s:.1f} | {row.avg_service_s:.1f} | "
            f"{row.avg_completion_s:.1f} | {row.ready_queue_avg:.1f}/{row.ready_queue_peak} | "
            f"{row.device_busy_pct:.1f}% |"
        )
    return "\n".join(lines)


def compact_lines(rows: Sequence[RunMetrics]) -> str:
    lines: list[str] = []
    for row in rows:
        name = DISPLAY_NAMES.get(row.scheduler, row.scheduler.upper())
        lines.append(
            f"{name} done={row.completed}/{row.count} ok={row.success_rate * 100:.1f}% "
            f"makespan={row.makespan_s:.1f}s arrival_end={row.arrival_end_s:.1f}s "
            f"drain={row.drain_tail_s:.1f}s tokens/s={row.total_tokens_per_s:.1f} "
            f"wait={row.avg_wait_s:.1f}s service={row.avg_service_s:.1f}s "
            f"completion={row.avg_completion_s:.1f}s ready={row.ready_queue_avg:.1f}/{row.ready_queue_peak} "
            f"device_busy={row.device_busy_pct:.1f}%"
        )
    return "\n".join(lines)


def write_outputs(output_root: Path, rows: Sequence[RunMetrics], prefix: str = "final_brief") -> None:
    table = markdown_table(rows)
    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    markdown = "\n".join(
        [
            "# Scheduler Run Brief",
            "",
            f"Generated: `{generated}`",
            "",
            table,
            "",
            "Definitions: arrival end is the last arrival relative to the first arrival; drain is "
            "makespan minus arrival end; device busy is the mean scheduling-layer busy share "
            "across workers.",
            "",
        ]
    )
    (output_root / f"{prefix}.md").write_text(markdown, encoding="utf-8")
    (output_root / f"{prefix}.txt").write_text(
        "MFE_FINAL_BRIEF_START\n" + compact_lines(rows) + "\nMFE_FINAL_BRIEF_END\n",
        encoding="utf-8",
    )

    fieldnames = list(asdict(rows[0]).keys())
    with (output_root / f"{prefix}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--schedulers",
        nargs="+",
        default=["fcfs", "sjf", "rhsail"],
        help="scheduler subdirectories in report order",
    )
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--prefix", default="final_brief")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    rows = collect_metrics(
        output_root,
        [scheduler.lower() for scheduler in args.schedulers],
        expected_count=args.expected_count,
        require_complete=not args.allow_incomplete,
    )
    write_outputs(output_root, rows, prefix=args.prefix)
    print("MFE_FINAL_BRIEF_START", flush=True)
    print(compact_lines(rows), flush=True)
    print("MFE_FINAL_BRIEF_END", flush=True)
    print(f"saved: {output_root / (args.prefix + '.md')}", flush=True)


if __name__ == "__main__":
    main()
