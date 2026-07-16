#!/usr/bin/env python3
"""Build a detailed report from completed scheduler experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
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
class DetailedRunMetrics:
    scheduler: str
    count: int
    completed: int
    success_rate: float
    makespan_s: float
    arrival_end_s: float
    drain_tail_s: float
    input_tokens: int
    output_tokens: int
    input_tokens_per_s: float
    output_tokens_per_s: float
    total_tokens_per_s: float
    avg_wait_s: float
    avg_run_time_s: float
    avg_service_s: float
    p99_service_s: float
    max_service_s: float
    avg_completion_s: float
    p99_completion_s: float
    max_completion_s: float
    p95_max_gap_s: float
    scheduler_overhead_s: float
    scheduler_overhead_pct: float
    ready_queue_avg: float
    ready_queue_peak: int
    device_busy_pct: float


@dataclass(frozen=True)
class DatasetMetrics:
    dataset: str
    scheduler: str
    count: int
    avg_service_s: float
    avg_run_time_s: float


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


def _numbers(values: Iterable[Any], label: str) -> list[float]:
    result = [float(value) for value in values if value is not None]
    if not result:
        raise ValueError(f"cannot compute {label} from an empty metric")
    return result


def percentile(values: Iterable[Any], q: float) -> float:
    """Return a NumPy-compatible linear percentile using only the standard library."""
    ordered = sorted(_numbers(values, f"p{q:g}"))
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"percentile must be in [0, 100], got {q}")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def request_max_gap(row: Mapping[str, Any]) -> float:
    intervals: list[tuple[float, float]] = []
    benchmark = row.get("benchmark") or {}
    if not isinstance(benchmark, Mapping):
        return 0.0
    for span in benchmark.values():
        if isinstance(span, (list, tuple)) and len(span) >= 2:
            start, end = float(span[0]), float(span[1])
            if end > start:
                intervals.append((start, end))
    merged = merge_intervals(intervals)
    return max(
        (merged[index + 1][0] - merged[index][1] for index in range(len(merged) - 1)),
        default=0.0,
    )


def load_run(
    run_dir: Path,
    scheduler: str,
    expected_count: int | None,
) -> tuple[DetailedRunMetrics, list[DatasetMetrics]]:
    summary_path = _single_file(run_dir, "*_summary.json", "summary JSON")
    detail_path = summary_path.with_name(
        summary_path.name.removesuffix("_summary.json") + ".json"
    )
    if not detail_path.is_file():
        raise ValueError(f"detail JSON matching {summary_path.name} is missing: {detail_path}")

    summary = _load_json(summary_path)
    details = _load_json(detail_path)
    if not isinstance(summary, Mapping):
        raise ValueError(f"summary JSON is not an object: {summary_path}")
    if not isinstance(details, list):
        raise ValueError(f"detail JSON is not a list: {detail_path}")

    count = int(summary.get("count") or 0)
    completed = int(summary.get("completed") or 0)
    success_rate = float(summary.get("success_rate") or 0.0)
    if expected_count is not None and count != expected_count:
        raise ValueError(f"{scheduler}: count {count}, expected {expected_count}")
    if len(details) != count:
        raise ValueError(f"{scheduler}: detail rows {len(details)}, summary count {count}")
    if completed != count or success_rate < 1.0 - 1e-9:
        raise ValueError(
            f"{scheduler}: incomplete run, completed={completed}/{count}, "
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
        raise ValueError(f"{scheduler}: valid detail rows {len(valid)}, completed {completed}")

    arrivals = _numbers((row.get("arrive_time") for row in valid), "arrival times")
    done_times = _numbers((row.get("done_time") for row in valid), "done times")
    waits = _numbers((row.get("idle_time") for row in valid), "waiting times")
    run_times = _numbers((row.get("run_time") for row in valid), "run times")
    services = _numbers((row.get("service_time") for row in valid), "service times")
    completions = _numbers((row.get("latency") for row in valid), "completion times")
    max_gaps = [request_max_gap(row) for row in valid]

    first_arrival = min(arrivals)
    arrival_end_s = max(arrivals) - first_arrival
    derived_makespan_s = max(done_times) - first_arrival
    makespan_s = float(summary.get("makespan") or derived_makespan_s)
    tolerance = max(1e-3, derived_makespan_s * 1e-6)
    if abs(makespan_s - derived_makespan_s) > tolerance:
        raise ValueError(
            f"{scheduler}: summary makespan {makespan_s:.6f} differs from "
            f"detail-derived makespan {derived_makespan_s:.6f}"
        )

    busy = summary.get("device_busy_pct") or {}
    if not isinstance(busy, Mapping) or not busy:
        raise ValueError(f"{scheduler}: device_busy_pct is missing")

    metrics = DetailedRunMetrics(
        scheduler=scheduler,
        count=count,
        completed=completed,
        success_rate=success_rate,
        makespan_s=makespan_s,
        arrival_end_s=arrival_end_s,
        drain_tail_s=max(0.0, makespan_s - arrival_end_s),
        input_tokens=int(summary.get("input_tokens") or 0),
        output_tokens=int(summary.get("output_tokens") or 0),
        input_tokens_per_s=float(summary.get("input_token_throughput") or 0.0),
        output_tokens_per_s=float(summary.get("output_token_throughput") or 0.0),
        total_tokens_per_s=float(summary.get("total_token_throughput") or 0.0),
        avg_wait_s=mean(waits),
        avg_run_time_s=mean(run_times),
        avg_service_s=mean(services),
        p99_service_s=percentile(services, 99),
        max_service_s=max(services),
        avg_completion_s=mean(completions),
        p99_completion_s=percentile(completions, 99),
        max_completion_s=max(completions),
        p95_max_gap_s=percentile(max_gaps, 95),
        scheduler_overhead_s=float(summary.get("scheduler_overhead_seconds") or 0.0),
        scheduler_overhead_pct=float(summary.get("scheduler_overhead_pct") or 0.0) * 100.0,
        ready_queue_avg=float(summary.get("ready_queue_avg") or 0.0),
        ready_queue_peak=int(summary.get("ready_queue_peak") or 0),
        device_busy_pct=mean(float(value) for value in busy.values()) * 100.0,
    )

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in valid:
        dataset = row.get("dataset")
        if dataset is None:
            raise ValueError(f"{scheduler}: detail row is missing dataset")
        grouped[str(dataset)].append(row)
    dataset_rows = [
        DatasetMetrics(
            dataset=dataset,
            scheduler=scheduler,
            count=len(items),
            avg_service_s=mean(float(item["service_time"]) for item in items),
            avg_run_time_s=mean(float(item["run_time"]) for item in items),
        )
        for dataset, items in sorted(grouped.items())
    ]
    return metrics, dataset_rows


def collect_metrics(
    output_root: Path,
    schedulers: Sequence[str],
    expected_count: int | None,
) -> tuple[list[DetailedRunMetrics], list[DatasetMetrics]]:
    overall: list[DetailedRunMetrics] = []
    datasets: list[DatasetMetrics] = []
    for scheduler in schedulers:
        run_dir = output_root / scheduler
        if not run_dir.is_dir():
            raise ValueError(f"scheduler output directory does not exist: {run_dir}")
        run_metrics, dataset_metrics = load_run(run_dir, scheduler, expected_count)
        overall.append(run_metrics)
        datasets.extend(dataset_metrics)
    return overall, datasets


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---:" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_markdown(
    overall: Sequence[DetailedRunMetrics],
    datasets: Sequence[DatasetMetrics],
) -> str:
    throughput_rows = []
    latency_rows = []
    for row in overall:
        name = DISPLAY_NAMES.get(row.scheduler, row.scheduler.upper())
        throughput_rows.append(
            [
                name,
                f"{row.completed}/{row.count}",
                f"{row.makespan_s:.1f}",
                f"{row.arrival_end_s:.1f}",
                f"{row.drain_tail_s:.1f}",
                f"{row.input_tokens_per_s:.1f}",
                f"{row.output_tokens_per_s:.1f}",
                f"{row.total_tokens_per_s:.1f}",
                f"{row.ready_queue_avg:.1f}/{row.ready_queue_peak}",
                f"{row.device_busy_pct:.1f}%",
                f"{row.scheduler_overhead_s:.1f}s / {row.scheduler_overhead_pct:.2f}%",
            ]
        )
        latency_rows.append(
            [
                name,
                f"{row.avg_wait_s:.1f}",
                f"{row.avg_run_time_s:.1f}",
                f"{row.avg_service_s:.1f}",
                f"{row.p99_service_s:.1f}",
                f"{row.max_service_s:.1f}",
                f"{row.avg_completion_s:.1f}",
                f"{row.p99_completion_s:.1f}",
                f"{row.max_completion_s:.1f}",
                f"{row.p95_max_gap_s:.1f}",
            ]
        )

    by_key = {(row.dataset, row.scheduler): row for row in datasets}
    dataset_rows = []
    for dataset in sorted({row.dataset for row in datasets}):
        values = [dataset]
        count = 0
        for overall_row in overall:
            item = by_key.get((dataset, overall_row.scheduler))
            if item is None:
                values.append("-")
            else:
                count = max(count, item.count)
                values.append(f"{item.avg_service_s:.1f} / {item.avg_run_time_s:.1f}")
        dataset_rows.append([dataset, str(count), *values[1:]])

    generated = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    return "\n".join(
        [
            "# Detailed Scheduler Run Brief",
            "",
            f"Generated: `{generated}`",
            "",
            "## System And Throughput",
            "",
            _markdown_table(
                [
                    "Strategy",
                    "Done",
                    "Makespan(s)",
                    "Arrival end(s)",
                    "Drain(s)",
                    "Input tok/s",
                    "Output tok/s",
                    "Total tok/s",
                    "Ready avg/peak",
                    "Device busy",
                    "Scheduler overhead",
                ],
                throughput_rows,
            ),
            "",
            "## Request Time And Continuity",
            "",
            _markdown_table(
                [
                    "Strategy",
                    "Avg wait(s)",
                    "Avg run(s)",
                    "Avg service(s)",
                    "P99 service(s)",
                    "Max service(s)",
                    "Avg completion(s)",
                    "P99 completion(s)",
                    "Max completion(s)",
                    "P95 max gap(s)",
                ],
                latency_rows,
            ),
            "",
            "## Dataset Average Service And Run Time",
            "",
            _markdown_table(
                [
                    "Dataset",
                    "Count/strategy",
                    *[
                        f"{DISPLAY_NAMES.get(row.scheduler, row.scheduler.upper())} service/run(s)"
                        for row in overall
                    ],
                ],
                dataset_rows,
            ),
            "",
            "Definitions: run time is the sum of all op durations for one request and excludes "
            "queue waiting and inter-op gaps; parallel op durations are counted separately, so "
            "run time may exceed the service window. Service is from the first op start until "
            "request completion; completion is from arrival until completion. P95 max gap is the 95th "
            "percentile across requests of each request's largest gap between merged op-active "
            "intervals. Scheduler overhead is cumulative scheduler decision time and its share "
            "of makespan.",
            "",
        ]
    )


def compact_lines(
    overall: Sequence[DetailedRunMetrics],
    datasets: Sequence[DatasetMetrics],
) -> str:
    lines = []
    for row in overall:
        name = DISPLAY_NAMES.get(row.scheduler, row.scheduler.upper())
        lines.append(
            f"{name} done={row.completed}/{row.count} makespan={row.makespan_s:.1f}s "
            f"arrival_end={row.arrival_end_s:.1f}s drain={row.drain_tail_s:.1f}s "
            f"input_tok/s={row.input_tokens_per_s:.1f} output_tok/s={row.output_tokens_per_s:.1f} "
            f"avg_wait/run/service={row.avg_wait_s:.1f}/{row.avg_run_time_s:.1f}/{row.avg_service_s:.1f}s "
            f"p99/max_service={row.p99_service_s:.1f}/{row.max_service_s:.1f}s "
            f"p99/max_completion={row.p99_completion_s:.1f}/{row.max_completion_s:.1f}s "
            f"p95_max_gap={row.p95_max_gap_s:.1f}s "
            f"scheduler_overhead={row.scheduler_overhead_s:.1f}s/{row.scheduler_overhead_pct:.2f}% "
            f"ready={row.ready_queue_avg:.1f}/{row.ready_queue_peak} "
            f"device_busy={row.device_busy_pct:.1f}%"
        )
    by_key = {(row.dataset, row.scheduler): row for row in datasets}
    for dataset in sorted({row.dataset for row in datasets}):
        values = []
        for run in overall:
            item = by_key.get((dataset, run.scheduler))
            name = DISPLAY_NAMES.get(run.scheduler, run.scheduler.upper())
            values.append(
                f"{name}={item.avg_service_s:.1f}/{item.avg_run_time_s:.1f}s"
                if item is not None
                else f"{name}=-"
            )
        lines.append(f"DATASET {dataset} avg_service/run " + " ".join(values))
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[Any]) -> None:
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_outputs(
    output_root: Path,
    overall: Sequence[DetailedRunMetrics],
    datasets: Sequence[DatasetMetrics],
    prefix: str,
) -> None:
    (output_root / f"{prefix}.md").write_text(
        build_markdown(overall, datasets), encoding="utf-8"
    )
    (output_root / f"{prefix}.txt").write_text(
        "MFE_DETAILED_BRIEF_START\n"
        + compact_lines(overall, datasets)
        + "\nMFE_DETAILED_BRIEF_END\n",
        encoding="utf-8",
    )
    write_csv(output_root / f"{prefix}_overall.csv", overall)
    write_csv(output_root / f"{prefix}_by_dataset.csv", datasets)


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
    parser.add_argument("--prefix", default="detailed_brief")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    overall, datasets = collect_metrics(
        output_root,
        [scheduler.lower() for scheduler in args.schedulers],
        args.expected_count,
    )
    write_outputs(output_root, overall, datasets, args.prefix)
    print("MFE_DETAILED_BRIEF_START", flush=True)
    print(compact_lines(overall, datasets), flush=True)
    print("MFE_DETAILED_BRIEF_END", flush=True)
    print(f"saved: {output_root / (args.prefix + '.md')}", flush=True)


if __name__ == "__main__":
    main()
