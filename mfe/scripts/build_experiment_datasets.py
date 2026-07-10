#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build fixed JSONL workloads for MFE DAG scheduling experiments."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from mfe.parser import build_ops_from_config, load_config
from mfe.scripts.process_datasets import PROCESSORS, _to_json_safe


DATASET_DAG_MAP: Dict[str, str] = {
    "gsm8k": "chain_gsm8k",
    "strategyqa": "branch_verify_strategyqa",
    "mmlu_pro": "debate_mmlu_pro",
    "math": "self_refine_math",
    "mbpp": "plan_code_test_mbpp",
    "hotpotqa": "parallel_debate_mapreduce_hotpotqa",
    "gpqa_diamond": "research_panel_gpqa_diamond",
    "swebench_verified": "agentic_repair_swebench_verified",
    "drop": "fork_join_k",
}

DEFAULT_YAML_BY_FAMILY: Dict[str, str] = {
    "chain_gsm8k": "bench/chain_gsm8k_medium.yaml",
    "branch_verify_strategyqa": "bench/branch_verify_strategyqa_medium.yaml",
    "debate_mmlu_pro": "bench/debate_mmlu_pro_medium.yaml",
    "self_refine_math": "bench/self_refine_math_medium.yaml",
    "plan_code_test_mbpp": "bench/plan_code_test_mbpp_medium.yaml",
    "parallel_debate_mapreduce_hotpotqa": "bench/parallel_debate_mapreduce_hotpotqa_medium.yaml",
    "research_panel_gpqa_diamond": "bench/research_panel_gpqa_diamond_medium.yaml",
    "agentic_repair_swebench_verified": "bench/agentic_repair_swebench_verified_medium.yaml",
    "path_n": "bench/path_4_medium.yaml",
    "unrolled_reflect_r": "bench/reflect_2_medium.yaml",
    "fork_join_k": "bench/fork_join_4_medium.yaml",
    "tree_reduce_kd": "bench/tree_reduce_2x2_medium.yaml",
    "debate": "bench/tournament_debate_4_medium.yaml",
    "diamond": "bench/diamond_medium.yaml",
    "plan_code_test": "bench/plan_code_test_medium.yaml",
    "large_mixed": "bench/large_mixed_medium.yaml",
}

SIZE_PRESETS = {
    "tiny": 1,
    "smoke": 5,
    "first50": 50,
    "first100": 100,
    "first200": 200,
    "first500": 500,
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
            line = json.dumps(_to_json_safe(row), ensure_ascii=False, sort_keys=True)
            # JSON permits U+2028/U+2029 inside strings, but many line-oriented
            # tools treat them as hard line breaks and corrupt JSONL records.
            line = line.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
            f.write(line + "\n")
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


class PromptLengthFilter:
    """Filter candidate records before sampling a fixed workload."""

    def __init__(
        self,
        *,
        templates_dir: str,
        tokenizer_path: str | None,
        prompt_token_limit: int | None,
        prompt_plus_max_tokens_limit: int | None,
    ) -> None:
        self.templates_dir = os.path.abspath(templates_dir)
        self.tokenizer_path = tokenizer_path
        self.prompt_token_limit = prompt_token_limit
        self.prompt_plus_max_tokens_limit = prompt_plus_max_tokens_limit
        self.enabled = prompt_token_limit is not None or prompt_plus_max_tokens_limit is not None
        self.tokenizer = None
        self._start_op_cache: Dict[str, List[Any]] = {}

        if tokenizer_path:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        elif self.enabled:
            print(
                "warning: prompt length filter enabled without --tokenizer-path; "
                "falling back to char/4 token estimates",
                flush=True,
            )

    def start_ops(self, yaml_name: str) -> List[Any]:
        if yaml_name not in self._start_op_cache:
            config_path = os.path.join(self.templates_dir, yaml_name)
            config = load_config(config_path)
            _, start_ops, _, _ = build_ops_from_config(config)
            self._start_op_cache[yaml_name] = start_ops
        return self._start_op_cache[yaml_name]

    def prompt_tokens(self, *, system_prompt: str, user_prompt: str) -> int:
        if self.tokenizer is None:
            header = system_prompt.strip()
            text = f"{header}\n\n{user_prompt}" if header else user_prompt
            return estimate_tokens(text)

        if hasattr(self.tokenizer, "apply_chat_template"):
            rendered = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": user_prompt or ""},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            header = system_prompt.strip()
            rendered = f"{header}\n\n{user_prompt}" if header else user_prompt
        return len(self.tokenizer.encode(rendered, add_special_tokens=False))

    def evaluate(self, record: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return {}

        prompt = str(record.get("question", "") or "")
        yaml_name = str(record.get("yaml", "") or "")
        start_stats: List[Tuple[str, int, int, int]] = []
        for op in self.start_ops(yaml_name):
            cfg = getattr(op, "model_config", None)
            system_prompt = str(getattr(cfg, "system_prompt", "") or "")
            op_max_tokens = int(getattr(cfg, "max_tokens", 0) or 0)
            tokens = self.prompt_tokens(system_prompt=system_prompt, user_prompt=prompt)
            start_stats.append((str(op.id), tokens, op_max_tokens, tokens + op_max_tokens))

        if not start_stats:
            return {
                "prompt_tokens_start_max": 0,
                "prompt_plus_start_op_max_tokens": 0,
                "start_op_max_tokens": 0,
                "start_ops": [],
            }

        return {
            "prompt_tokens_start_max": max(item[1] for item in start_stats),
            "prompt_plus_start_op_max_tokens": max(item[3] for item in start_stats),
            "start_op_max_tokens": max(item[2] for item in start_stats),
            "start_ops": [item[0] for item in start_stats],
        }

    def apply(self, record: Dict[str, Any]) -> Tuple[bool, str | None]:
        stats = self.evaluate(record)
        if stats:
            record.update(stats)

        if not self.enabled:
            return True, None

        prompt_tokens = int(stats.get("prompt_tokens_start_max", 0))
        prompt_plus_max = int(stats.get("prompt_plus_start_op_max_tokens", 0))
        if self.prompt_token_limit is not None and prompt_tokens >= self.prompt_token_limit:
            return False, "prompt_token_limit"
        if self.prompt_plus_max_tokens_limit is not None and prompt_plus_max > self.prompt_plus_max_tokens_limit:
            return False, "prompt_plus_max_tokens_limit"
        return True, None

    def manifest(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "tokenizer_path": self.tokenizer_path,
            "templates_dir": self.templates_dir,
            "prompt_token_limit_exclusive": self.prompt_token_limit,
            "prompt_plus_max_tokens_limit_inclusive": self.prompt_plus_max_tokens_limit,
            "token_counter": "tokenizer" if self.tokenizer is not None else "estimate_chars_div_4",
        }


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
    length_filter = PromptLengthFilter(
        templates_dir=args.templates_dir,
        tokenizer_path=args.tokenizer_path,
        prompt_token_limit=args.prompt_token_limit,
        prompt_plus_max_tokens_limit=args.prompt_plus_max_tokens_limit,
    )
    load_limit = args.load_limit
    if load_limit is None and not length_filter.enabled:
        load_limit = max(per_dataset * 4, per_dataset)

    all_records: List[Dict[str, Any]] = []
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    candidate_counts: Dict[str, int] = {}
    eligible_counts: Dict[str, int] = {}
    filtered_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for dataset in args.datasets:
        dataset = dataset.lower()
        rows = load_dataset_rows(data_dir, dataset, load_limit)
        rows = [r for r in rows if str(r.get("question", "") or "").strip()]
        candidate_counts[dataset] = len(rows)
        candidates: List[Dict[str, Any]] = []
        for i, row in enumerate(rows):
            record = make_record(row, dataset=dataset, index=i, output_length=args.output_length)
            keep, reason = length_filter.apply(record)
            if keep:
                candidates.append(record)
            else:
                filtered_counts[dataset][str(reason)] += 1
        eligible_counts[dataset] = len(candidates)
        if len(candidates) < per_dataset:
            print(
                f"{dataset}: only {len(candidates)} eligible rows after filters, "
                f"skip size={args.size} need={per_dataset} candidates={len(rows)}"
            )
            continue
        if args.selection == "random":
            rng.shuffle(candidates)
        records = candidates[:per_dataset]
        by_dataset[dataset] = records
        all_records.extend(records)
        path = os.path.join(out_dir, f"{dataset}_{args.output_length}_{args.size}.jsonl")
        print(
            f"{dataset}: {write_jsonl(path, records)} rows -> {path} "
            f"(eligible={len(candidates)} candidates={len(rows)} filtered={sum(filtered_counts[dataset].values())})"
        )

    rng.shuffle(all_records)
    mixed_path = os.path.join(out_dir, f"mixed_{args.output_length}_{args.size}.jsonl")
    print(f"mixed: {write_jsonl(mixed_path, all_records)} rows -> {mixed_path}")

    manifest = {
        "seed": args.seed,
        "size": args.size,
        "per_dataset": per_dataset,
        "output_length": args.output_length,
        "mixed_order": "global_random_shuffle",
        "datasets": args.datasets,
        "counts": {k: len(v) for k, v in by_dataset.items()},
        "candidate_counts": candidate_counts,
        "eligible_counts": eligible_counts,
        "filtered_counts": {k: dict(v) for k, v in filtered_counts.items()},
        "length_filter": length_filter.manifest(),
        "skipped": {
            dataset: f"fewer than {per_dataset} eligible rows"
            for dataset in args.datasets
            if dataset.lower() not in by_dataset
        },
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
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["gsm8k", "strategyqa", "mmlu_pro", "math", "mbpp", "hotpotqa", "swebench_verified"],
    )
    p.add_argument("--size", choices=sorted(SIZE_PRESETS), default="smoke")
    p.add_argument("--per-dataset", type=int, default=None)
    p.add_argument("--load-limit", type=int, default=None)
    p.add_argument("--output-length", choices=("short", "medium", "long"), default="medium")
    p.add_argument("--selection", choices=("first", "random"), default="random", help="select first N rows or random N rows per dataset before global shuffle")
    p.add_argument("--seed", type=int, default=20260707)
    p.add_argument("--templates-dir", default="templates", help="template root used for prompt-length filtering")
    p.add_argument("--tokenizer-path", default=None, help="Hugging Face tokenizer/model path for exact prompt token filtering")
    p.add_argument("--prompt-token-limit", type=int, default=None, help="drop candidates with start prompt tokens >= this limit")
    p.add_argument(
        "--prompt-plus-max-tokens-limit",
        type=int,
        default=None,
        help="drop candidates with start prompt tokens plus start-op max_tokens above this limit",
    )
    p.add_argument("--builtin-tiny", action="store_true", help="build a small built-in workload without external datasets")
    args = p.parse_args()
    if args.builtin_tiny:
        build_builtin_tiny(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
