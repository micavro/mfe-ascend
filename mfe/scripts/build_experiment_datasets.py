#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build fixed JSONL workloads for MFE DAG scheduling experiments."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List

from mfe.scripts.process_datasets import PROCESSORS


DATASET_DAG_MAP: Dict[str, str] = {
    "gsm8k": "path_n",
    "math": "unrolled_reflect_r",
    "drop": "fork_join_k",
    "hotpotqa": "tree_reduce_kd",
    "strategyqa": "diamond",
    "mbpp": "plan_code_test",
}

DEFAULT_YAML_BY_FAMILY: Dict[str, str] = {
    "path_n": "bench/path_4_medium.yaml",
    "unrolled_reflect_r": "bench/reflect_2_medium.yaml",
    "fork_join_k": "bench/fork_join_4_medium.yaml",
    "tree_reduce_kd": "bench/tree_reduce_2x2_medium.yaml",
    "diamond": "bench/diamond_medium.yaml",
    "plan_code_test": "bench/plan_code_test_medium.yaml",
    "large_mixed": "bench/large_mixed_medium.yaml",
}

SIZE_PRESETS = {
    "smoke": 5,
    "dev": 25,
    "full": 100,
}

LENGTH_SUFFIX = {
    "short": "short",
    "medium": "medium",
    "long": "long",
}


def estimate_tokens(text: Any) -> int:
    s = str(text or "")
    if not s:
        return 0
    return max(1, (len(s) + 3) // 4)


def normalize_answer(row: Dict[str, Any]) -> Any:
    for key in ("answer", "answers", "solution", "code", "canonical_solution", "target", "label"):
        if key in row and row[key] not in (None, ""):
            return row[key]
    return ""


def yaml_for_family(family: str, output_length: str) -> str:
    return DEFAULT_YAML_BY_FAMILY[family]


def make_record(
    row: Dict[str, Any],
    *,
    dataset: str,
    index: int,
    output_length: str,
    dag_family: str | None = None,
    yaml_name: str | None = None,
) -> Dict[str, Any]:
    family = dag_family or DATASET_DAG_MAP.get(dataset, "path_n")
    question = str(row.get("question", "") or "").strip()
    if not question:
        question = str(row.get("question_short", row.get("prompt", row.get("problem", ""))) or "").strip()
    selected_yaml = yaml_name or yaml_for_family(family, output_length)
    metadata = {
        k: v
        for k, v in row.items()
        if k not in {"question", "answer", "answers", "solution", "code", "canonical_solution"}
    }
    return {
        "sample_id": f"{dataset}-{index:06d}",
        "dataset": dataset,
        "dag_family": family,
        "yaml": selected_yaml,
        "question": question,
        "answer": normalize_answer(row),
        "input_len_chars": len(question),
        "input_len_est_tokens": estimate_tokens(question),
        "output_len_target": output_length,
        "metadata": metadata,
    }


def load_dataset_rows(data_dir: str, dataset: str, limit: int | None) -> List[Dict[str, Any]]:
    dataset = dataset.lower()
    if dataset not in PROCESSORS:
        raise ValueError(f"unknown dataset {dataset}; available={sorted(PROCESSORS)}")
    return PROCESSORS[dataset](data_dir, limit)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.output_dir)
    per_dataset = args.per_dataset or SIZE_PRESETS[args.size]
    load_limit = args.load_limit or max(per_dataset * 4, per_dataset)

    all_records: List[Dict[str, Any]] = []
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for dataset in args.datasets:
        dataset = dataset.lower()
        rows = load_dataset_rows(data_dir, dataset, load_limit)
        rows = [r for r in rows if str(r.get("question", "") or "").strip()]
        rng.shuffle(rows)
        selected = rows[:per_dataset]
        records = [
            make_record(r, dataset=dataset, index=i, output_length=args.output_length)
            for i, r in enumerate(selected)
        ]
        by_dataset[dataset] = records
        all_records.extend(records)
        path = os.path.join(out_dir, f"{dataset}_{args.output_length}_{args.size}.jsonl")
        print(f"{dataset}: {write_jsonl(path, records)} rows -> {path}")

    rng.shuffle(all_records)
    mixed_path = os.path.join(out_dir, f"mixed_{args.output_length}_{args.size}.jsonl")
    print(f"mixed: {write_jsonl(mixed_path, all_records)} rows -> {mixed_path}")

    manifest = {
        "seed": args.seed,
        "size": args.size,
        "per_dataset": per_dataset,
        "output_length": args.output_length,
        "datasets": args.datasets,
        "counts": {k: len(v) for k, v in by_dataset.items()},
        "mixed_count": len(all_records),
    }
    manifest_path = os.path.join(out_dir, f"manifest_{args.output_length}_{args.size}.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"manifest -> {manifest_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build fixed JSONL workloads for MFE experiments")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="data/experiments")
    p.add_argument("--datasets", nargs="+", default=["gsm8k", "math", "drop", "hotpotqa", "strategyqa", "mbpp"])
    p.add_argument("--size", choices=sorted(SIZE_PRESETS), default="smoke")
    p.add_argument("--per-dataset", type=int, default=None)
    p.add_argument("--load-limit", type=int, default=None)
    p.add_argument("--output-length", choices=("short", "medium", "long"), default="medium")
    p.add_argument("--seed", type=int, default=20260707)
    args = p.parse_args()
    build(args)


if __name__ == "__main__":
    main()
