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
    "tiny": 1,
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


BUILTIN_TINY_ROWS: List[Dict[str, str]] = [
    {
        "dataset": "builtin_math",
        "dag_family": "path_n",
        "question": "A store sells 18 notebooks in the morning and 27 in the afternoon. If each notebook costs 3 dollars, what is the total revenue?",
        "answer": "135",
    },
    {
        "dataset": "builtin_math",
        "dag_family": "unrolled_reflect_r",
        "question": "Solve for x: 3x + 14 = 65. Return the value of x.",
        "answer": "17",
    },
    {
        "dataset": "builtin_reading",
        "dag_family": "fork_join_k",
        "question": "Passage: Mina tested three batteries. Battery A lasted 6 hours, Battery B lasted 9 hours, and Battery C lasted 7 hours. Question: Which battery lasted the longest and by how many hours more than Battery A?",
        "answer": "Battery B, 3 hours more than Battery A.",
    },
    {
        "dataset": "builtin_multihop",
        "dag_family": "tree_reduce_kd",
        "question": "Alice is Bob's sister. Bob is Carol's father. Carol has a brother named Dan. What is Alice's family relationship to Dan?",
        "answer": "Alice is Dan's aunt.",
    },
    {
        "dataset": "builtin_strategy",
        "dag_family": "diamond",
        "question": "Could a person carry a bicycle through a standard doorway without disassembling it? Explain briefly before the final answer.",
        "answer": "Yes.",
    },
    {
        "dataset": "builtin_code",
        "dag_family": "plan_code_test",
        "question": "Write a Python function add_even_numbers(nums) that returns the sum of all even integers in nums.",
        "answer": "def add_even_numbers(nums): return sum(x for x in nums if x % 2 == 0)",
    },
    {
        "dataset": "builtin_debate",
        "dag_family": "large_mixed",
        "question": "A team can choose Plan A with low risk and moderate reward or Plan B with high risk and high reward. Give a balanced recommendation assuming reliability matters most.",
        "answer": "Prefer Plan A unless the reward gap is mission critical.",
    },
]


def build_builtin_tiny(args: argparse.Namespace) -> None:
    out_dir = os.path.abspath(args.output_dir)
    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(BUILTIN_TINY_ROWS):
        family = item["dag_family"]
        rows.append(
            make_record(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "source": "builtin_tiny",
                },
                dataset=item["dataset"],
                index=index,
                output_length=args.output_length,
                dag_family=family,
                yaml_name=yaml_for_family(family, args.output_length),
            )
        )
    mixed_path = os.path.join(out_dir, f"mixed_{args.output_length}_tiny.jsonl")
    count = write_jsonl(mixed_path, rows)
    manifest = {
        "seed": None,
        "size": "tiny",
        "builtin_tiny": True,
        "output_length": args.output_length,
        "mixed_count": count,
        "datasets": sorted({row["dataset"] for row in rows}),
        "counts": {
            dataset: sum(1 for row in rows if row["dataset"] == dataset)
            for dataset in sorted({row["dataset"] for row in rows})
        },
    }
    manifest_path = os.path.join(out_dir, f"manifest_{args.output_length}_tiny.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"builtin tiny: {count} rows -> {mixed_path}")
    print(f"manifest -> {manifest_path}")


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
    p.add_argument("--builtin-tiny", action="store_true", help="build a small built-in workload without external datasets")
    args = p.parse_args()
    if args.builtin_tiny:
        build_builtin_tiny(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
