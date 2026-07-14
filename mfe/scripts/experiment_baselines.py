#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run FCFS/SJF baseline experiments on JSONL DAG workloads."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

_script_dir = os.path.dirname(os.path.abspath(__file__))
_package_root = os.path.dirname(_script_dir)
_project_root = os.path.dirname(_package_root)
sys.path.insert(0, _project_root)

from mfe.config import set_verbose
from mfe.parser import build_from_path
from mfe.serve import run_server
from mfe.scripts.client import Client, _json_default, _to_json_safe, _zero_timestamps, run_data_test
from mfe.runtime import RuntimeConfig, collect_run_info


OUTPUT_TOKENS = {
    "short": 256,
    "medium": 1024,
    "long": 4096,
}

BRIEF_FIELDS = [
    "scheduler",
    "output_length",
    "repeat_index",
    "count",
    "completed",
    "success_rate",
    "makespan_s",
    "total_tokens",
    "output_tokens",
    "total_tokens_per_s",
    "output_tokens_per_s",
    "req_per_s",
    "avg_wait_s",
    "p95_latency_s",
    "ready_queue_peak",
    "device_busy_pct",
    "load_imbalance",
    "parallelism_utilization",
]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    arr = sorted(values)
    if len(arr) == 1:
        return float(arr[0])
    pos = (len(arr) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(arr) - 1)
    frac = pos - lo
    return float(arr[lo] * (1 - frac) + arr[hi] * frac)


def numeric_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return mean(vals) if vals else None


def template_path(templates_dir: str, yaml_name: str) -> str:
    if os.path.isabs(yaml_name):
        return yaml_name
    return os.path.join(templates_dir, yaml_name)


def get_template_edges(templates_dir: str, yaml_name: str) -> List[Tuple[str, str]]:
    try:
        ops, _, _, _ = build_from_path(template_path(templates_dir, yaml_name))
    except Exception:
        return []
    edges: List[Tuple[str, str]] = []
    for op_id, op in ops.items():
        for child in getattr(op, "output_ops", []) or []:
            edges.append((op_id, str(child.id)))
    return edges


def critical_path_duration(templates_dir: str, yaml_name: str, durations: Mapping[str, float]) -> float:
    try:
        ops, _, end_ops, _ = build_from_path(template_path(templates_dir, yaml_name))
    except Exception:
        return 0.0
    memo: Dict[str, float] = {}

    def dfs(op_id: str) -> float:
        if op_id in memo:
            return memo[op_id]
        op = ops[op_id]
        own = float(durations.get(op_id, 0.0) or 0.0)
        if not getattr(op, "output_ops", None):
            memo[op_id] = own
            return own
        tail = max((dfs(str(child.id)) for child in op.output_ops if str(child.id) in ops), default=0.0)
        memo[op_id] = own + tail
        return memo[op_id]

    return max((dfs(op_id) for op_id in ops), default=0.0)


def dependency_stall_time(templates_dir: str, result: Mapping[str, Any]) -> float:
    bench = result.get("benchmark") or {}
    edges = get_template_edges(templates_dir, str(result.get("yaml") or ""))
    if not bench:
        return 0.0
    preds: Dict[str, List[str]] = defaultdict(list)
    for src, dst in edges:
        preds[dst].append(src)
    total = 0.0
    for op_id, parents in preds.items():
        if op_id not in bench:
            continue
        parent_ends = [float(bench[p][1]) for p in parents if p in bench and len(bench[p]) >= 2]
        if not parent_ends:
            continue
        total += max(0.0, float(bench[op_id][0]) - max(parent_ends))
    return total


def cross_device_dependencies(templates_dir: str, result: Mapping[str, Any]) -> int:
    assignments = result.get("worker_assignments") or {}
    total = 0
    for src, dst in get_template_edges(templates_dir, str(result.get("yaml") or "")):
        if src in assignments and dst in assignments and assignments[src] != assignments[dst]:
            total += 1
    return total


def token_totals(result: Mapping[str, Any]) -> Dict[str, int]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for metrics in (result.get("op_metrics") or {}).values():
        if not isinstance(metrics, Mapping):
            continue
        totals["input_tokens"] += int(metrics.get("input_tokens") or 0)
        totals["output_tokens"] += int(metrics.get("output_tokens") or 0)
        totals["total_tokens"] += int(metrics.get("total_tokens") or 0)
    if totals["total_tokens"] <= 0:
        est = int(result.get("input_len_est_tokens") or 0)
        totals["input_tokens"] = est
        totals["output_tokens"] = 0
        totals["total_tokens"] = est
    return totals


def summarize(results: List[Dict[str, Any]], *, templates_dir: str, device_count: int) -> Dict[str, Any]:
    valid = [r for r in results if r.get("status", "completed") == "completed" and r.get("done_time") is not None and r.get("arrive_time") is not None]
    if valid:
        makespan = max(float(r["done_time"]) for r in valid) - min(float(r["arrive_time"]) for r in valid)
    else:
        makespan = 0.0

    latencies = [float(r["latency"]) for r in valid if r.get("latency") is not None]
    waits = [float(r["idle_time"]) for r in valid if r.get("idle_time") is not None]
    services = [float(r["service_time"]) for r in valid if r.get("service_time") is not None]
    run_times = [float(r["run_time"]) for r in valid if r.get("run_time") is not None]

    token_rows = [token_totals(r) for r in valid]
    input_tokens = sum(r["input_tokens"] for r in token_rows)
    output_tokens = sum(r["output_tokens"] for r in token_rows)
    total_tokens = sum(r["total_tokens"] for r in token_rows)

    device_busy: Dict[str, float] = defaultdict(float)
    device_output_tokens: Dict[str, int] = defaultdict(int)
    for result in valid:
        durations = result.get("op_durations") or {}
        assignments = result.get("worker_assignments") or {}
        metrics = result.get("op_metrics") or {}
        for op_id, duration in durations.items():
            worker = str(assignments.get(op_id, "unknown"))
            device_busy[worker] += float(duration)
            if isinstance(metrics.get(op_id), Mapping):
                device_output_tokens[worker] += int(metrics[op_id].get("output_tokens") or 0)
    for worker_id in range(max(1, device_count)):
        device_busy.setdefault(str(worker_id), 0.0)
        device_output_tokens.setdefault(str(worker_id), 0)

    busy_values = list(device_busy.values())
    critical_paths = [critical_path_duration(templates_dir, str(r.get("yaml") or ""), r.get("op_durations") or {}) for r in valid]
    dependency_stalls = [dependency_stall_time(templates_dir, r) for r in valid]
    cross_edges = [cross_device_dependencies(templates_dir, r) for r in valid]
    scheduler_snapshots = [r.get("scheduler_metrics") or {} for r in valid if isinstance(r.get("scheduler_metrics"), Mapping)]
    scheduler_overhead = max((float(s.get("overhead_seconds") or 0.0) for s in scheduler_snapshots), default=0.0)

    by_family: Dict[str, Dict[str, Any]] = {}
    families = sorted({str(r.get("dag_family") or r.get("yaml") or "unknown") for r in valid})
    for family in families:
        rows = [r for r in valid if str(r.get("dag_family") or r.get("yaml") or "unknown") == family]
        fam_tokens = [token_totals(r) for r in rows]
        fam_makespan = 0.0
        if rows:
            fam_makespan = max(float(r["done_time"]) for r in rows) - min(float(r["arrive_time"]) for r in rows)
        by_family[family] = {
            "count": len(rows),
            "latency_mean": numeric_mean([r.get("latency") for r in rows]),
            "output_tokens": sum(x["output_tokens"] for x in fam_tokens),
            "token_throughput": (sum(x["total_tokens"] for x in fam_tokens) / fam_makespan if fam_makespan > 0 else 0.0),
        }

    return {
        "count": len(results),
        "completed": len(valid),
        "success_rate": (len(valid) / len(results) if results else 0.0),
        "makespan": makespan,
        "request_throughput": (len(valid) / makespan if makespan > 0 else 0.0),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "input_token_throughput": (input_tokens / makespan if makespan > 0 else 0.0),
        "output_token_throughput": (output_tokens / makespan if makespan > 0 else 0.0),
        "total_token_throughput": (total_tokens / makespan if makespan > 0 else 0.0),
        "goodput_tokens_per_second": (total_tokens / makespan if makespan > 0 else 0.0),
        "latency_mean": numeric_mean(latencies),
        "latency_p50": percentile(latencies, 0.50),
        "latency_p95": percentile(latencies, 0.95),
        "latency_p99": percentile(latencies, 0.99),
        "waiting_time_mean": numeric_mean(waits),
        "service_time_mean": numeric_mean(services),
        "run_time_mean": numeric_mean(run_times),
        "scheduler_overhead_seconds": scheduler_overhead,
        "scheduler_overhead_pct": (scheduler_overhead / makespan if makespan > 0 else 0.0),
        "ready_queue_avg": max((float(s.get("ready_queue_avg") or 0.0) for s in scheduler_snapshots), default=0.0),
        "ready_queue_peak": max((int(s.get("ready_queue_peak") or 0) for s in scheduler_snapshots), default=0),
        "critical_path_mean": numeric_mean(critical_paths),
        "dag_parallelism_mean": numeric_mean([
            (float(r.get("run_time") or 0.0) / cp if cp > 0 else None)
            for r, cp in zip(valid, critical_paths)
        ]),
        "dependency_stall_mean": numeric_mean(dependency_stalls),
        "cross_device_dependencies_mean": numeric_mean([float(x) for x in cross_edges]),
        "device_busy_seconds": dict(sorted(device_busy.items())),
        "device_busy_pct": {
            k: (v / makespan if makespan > 0 else 0.0)
            for k, v in sorted(device_busy.items())
        },
        "device_output_tokens_per_second": {
            k: (v / makespan if makespan > 0 else 0.0)
            for k, v in sorted(device_output_tokens.items())
        },
        "load_imbalance": (max(busy_values) / min([x for x in busy_values if x > 0]) if busy_values and any(x > 0 for x in busy_values) else 0.0),
        "parallelism_utilization": (sum(run_times) / (makespan * max(1, device_count)) if makespan > 0 else 0.0),
        "by_dag_family": by_family,
    }


def write_summary_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "scheduler",
        "repeat_index",
        "questions_file",
        "output_length",
        "count",
        "completed",
        "success_rate",
        "makespan",
        "request_throughput",
        "total_token_throughput",
        "output_token_throughput",
        "waiting_time_mean",
        "latency_p95",
        "scheduler_overhead_pct",
        "parallelism_utilization",
        "load_imbalance",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def compact_device_busy(summary: Mapping[str, Any]) -> str:
    busy = summary.get("device_busy_pct") or {}
    if not isinstance(busy, Mapping):
        return ""
    return ";".join(f"{k}:{float(v):.4f}" for k, v in sorted(busy.items(), key=lambda x: str(x[0])))


def brief_summary_row(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "scheduler": summary.get("scheduler"),
        "output_length": summary.get("output_length"),
        "repeat_index": summary.get("repeat_index"),
        "count": summary.get("count"),
        "completed": summary.get("completed"),
        "success_rate": round(float(summary.get("success_rate") or 0.0), 4),
        "makespan_s": round(float(summary.get("makespan") or 0.0), 4),
        "total_tokens": summary.get("total_tokens"),
        "output_tokens": summary.get("output_tokens"),
        "total_tokens_per_s": round(float(summary.get("total_token_throughput") or 0.0), 4),
        "output_tokens_per_s": round(float(summary.get("output_token_throughput") or 0.0), 4),
        "req_per_s": round(float(summary.get("request_throughput") or 0.0), 4),
        "avg_wait_s": round(float(summary.get("waiting_time_mean") or 0.0), 4),
        "p95_latency_s": round(float(summary.get("latency_p95") or 0.0), 4),
        "ready_queue_peak": summary.get("ready_queue_peak"),
        "device_busy_pct": compact_device_busy(summary),
        "load_imbalance": round(float(summary.get("load_imbalance") or 0.0), 4),
        "parallelism_utilization": round(float(summary.get("parallelism_utilization") or 0.0), 4),
    }


def print_brief_summary(summary: Mapping[str, Any]) -> None:
    row = brief_summary_row(summary)
    print("MFE_BRIEF_RESULT_START", flush=True)
    print(",".join(BRIEF_FIELDS), flush=True)
    print(",".join(str(row.get(field, "")) for field in BRIEF_FIELDS), flush=True)
    print("MFE_BRIEF_RESULT_END", flush=True)


def write_brief_outputs(output_dir: str, summaries: List[Dict[str, Any]]) -> None:
    rows = [brief_summary_row(summary) for summary in summaries]
    csv_path = os.path.join(output_dir, "brief_summary.csv")
    txt_path = os.path.join(output_dir, "brief_summary.txt")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BRIEF_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("MFE_BRIEF_RESULT_START\n")
        f.write(",".join(BRIEF_FIELDS) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(field, "")) for field in BRIEF_FIELDS) + "\n")
        f.write("MFE_BRIEF_RESULT_END\n")
    print(f"saved brief summary: {csv_path}")
    print(f"saved brief text: {txt_path}")


def run_once(args: argparse.Namespace, questions: List[Dict[str, Any]], repeat_index: int, run_info: Dict[str, Any]) -> Dict[str, Any]:
    output_max_tokens = int(args.output_max_tokens or OUTPUT_TOKENS[args.output_length])
    os.environ["MFE_SCHEDULER"] = args.scheduler
    os.environ["MFE_OUTPUT_MAX_TOKENS"] = str(output_max_tokens)
    if args.test_worker:
        os.environ["MFE_TEST_OUTPUT_TOKENS"] = str(output_max_tokens)
    if args.worker_delay is not None:
        os.environ["MFE_TEST_WORKER_DELAY"] = str(args.worker_delay)
    if args.verbose:
        os.environ["MFE_VERBOSE"] = "1"

    req_q = mp.Queue()
    resp_q = mp.Queue()
    proc = mp.Process(target=run_server, args=(req_q, resp_q, args.templates_dir, args.test_worker), daemon=False)
    proc.start()
    client = Client(req_q, resp_q)
    try:
        results = run_data_test(
            client,
            questions,
            send_interval=args.send_interval,
            arrival_mode=args.arrival_mode,
            arrival_batch_size=args.arrival_batch_size,
            poisson_rate=args.poisson_rate,
            arrival_seed=args.arrival_seed,
        )
    finally:
        client.close()
        proc.join(timeout=5.0)

    _zero_timestamps(results)
    for result in results:
        result["run_info"] = run_info
        result["scheduler"] = result.get("scheduler") or args.scheduler
        result["output_length"] = args.output_length
        result["repeat_index"] = repeat_index

    summary = summarize(results, templates_dir=args.templates_dir, device_count=args.device_count)
    summary.update({
        "scheduler": args.scheduler,
        "repeat_index": repeat_index,
        "questions_file": args.questions_file,
        "output_length": args.output_length,
        "output_max_tokens": output_max_tokens,
        "test_worker": bool(args.test_worker),
        "send_interval": args.send_interval,
        "arrival_mode": args.arrival_mode,
        "arrival_batch_size": args.arrival_batch_size,
        "poisson_rate": args.poisson_rate,
        "arrival_seed": args.arrival_seed,
        "device_count": args.device_count,
        "run_info": run_info,
    })

    base = f"{os.path.splitext(os.path.basename(args.questions_file))[0]}_{args.scheduler}_{args.output_length}_run{repeat_index}"
    detail_path = os.path.join(args.output_dir, f"{base}.json")
    summary_path = os.path.join(args.output_dir, f"{base}_summary.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(results), f, ensure_ascii=False, indent=2, default=_json_default)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(summary), f, ensure_ascii=False, indent=2, default=_json_default)
    print(f"saved detail: {detail_path}")
    print(f"saved summary: {summary_path}")
    print_brief_summary(summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Run MFE baseline scheduler experiments")
    p.add_argument("--questions-file", required=True)
    p.add_argument("--scheduler", choices=("fcfs", "sjf", "eager", "sailp", "darc", "rhsail"), default="fcfs")
    p.add_argument("--output-length", choices=sorted(OUTPUT_TOKENS), default="medium")
    p.add_argument("--output-max-tokens", type=int, default=None, help="override the max generated tokens for every DAG op")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--templates-dir", default="templates")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--send-interval", type=float, default=0.0)
    p.add_argument("--arrival-mode", choices=("fixed", "poisson-burst", "poisson-batch"), default="fixed")
    p.add_argument("--arrival-batch-size", type=int, default=None)
    p.add_argument("--poisson-rate", type=float, default=1.0, help="burst arrivals per second for --arrival-mode poisson-burst")
    p.add_argument("--arrival-seed", type=int, default=20260709)
    p.add_argument("--test-worker", action="store_true")
    p.add_argument("--worker-delay", type=float, default=None)
    p.add_argument("--device-count", type=int, default=None)
    p.add_argument("--accelerator", choices=("ascend", "cuda"), default=None)
    p.add_argument("--device-ids", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--offline", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    if args.output_max_tokens is not None and args.output_max_tokens <= 0:
        raise SystemExit("--output-max-tokens must be positive")
    if args.arrival_mode == "poisson-batch":
        args.arrival_mode = "poisson-burst"

    if args.verbose:
        set_verbose(True)

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(root)
    args.templates_dir = os.path.abspath(args.templates_dir)
    args.questions_file = os.path.abspath(args.questions_file)
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = os.path.join(root, "data", "experiments", "runs", args.scheduler, args.output_length, stamp)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    runtime = RuntimeConfig.from_values(
        accelerator=args.accelerator,
        device_ids=args.device_ids,
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        offline=args.offline or None,
        project_root=root,
    )
    runtime.apply()
    if args.device_count is None:
        if args.test_worker:
            args.device_count = int(os.environ.get("MFE_TEST_DEVICE_CNT", "4"))
        else:
            ids = [x for x in (runtime.device_ids or "").split(",") if x.strip()]
            args.device_count = len(ids) if ids else 1
    if args.arrival_batch_size is None and args.arrival_mode == "poisson-burst":
        args.arrival_batch_size = max(1, int(args.device_count or 1))

    questions = read_jsonl(args.questions_file)
    if not questions:
        raise RuntimeError(f"no questions found in {args.questions_file}")
    for row in questions:
        row.setdefault("input_len_chars", len(str(row.get("question", "") or "")))
        row.setdefault("input_len_est_tokens", max(1, (int(row["input_len_chars"]) + 3) // 4) if row["input_len_chars"] else 0)

    run_info = collect_run_info(runtime, cwd=root)
    summaries = [run_once(args, questions, i, run_info) for i in range(1, args.repeat + 1)]
    aggregate_path = os.path.join(args.output_dir, "summary_all.json")
    csv_path = os.path.join(args.output_dir, "summary_all.csv")
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(summaries), f, ensure_ascii=False, indent=2, default=_json_default)
    write_summary_csv(csv_path, summaries)
    write_brief_outputs(args.output_dir, summaries)
    print(f"saved aggregate summary: {aggregate_path}")
    print(f"saved csv summary: {csv_path}")


if __name__ == "__main__":
    main()
