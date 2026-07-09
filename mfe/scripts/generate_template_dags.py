#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate simple TikZ DAG diagrams for YAML workflow templates."""

from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import yaml


BENCH_BASIC_ORDER = [
    "bench/chain_gsm8k_medium.yaml",
    "bench/branch_verify_strategyqa_medium.yaml",
    "bench/debate_mmlu_pro_medium.yaml",
    "bench/self_refine_math_medium.yaml",
    "bench/plan_code_test_mbpp_medium.yaml",
]

BENCH_COMPLEX_ORDER = [
    "bench/research_panel_gpqa_diamond_medium.yaml",
    "bench/agentic_repair_swebench_verified_medium.yaml",
]

BENCH_ORDER = BENCH_BASIC_ORDER + BENCH_COMPLEX_ORDER
BENCH_HIGHLIGHTS = set(BENCH_ORDER)
BENCH_COMPLEX = set(BENCH_COMPLEX_ORDER)

BENCH_DAG_TYPES = {
    "bench/chain_gsm8k_medium.yaml": "线性链式推理",
    "bench/branch_verify_strategyqa_medium.yaml": "隐式分解 + 双分支校验",
    "bench/debate_mmlu_pro_medium.yaml": "多路并行辩论 + 裁决",
    "bench/self_refine_math_medium.yaml": "草稿 - 批改 - 修正",
    "bench/plan_code_test_mbpp_medium.yaml": "计划 - 代码 - 测试",
    "bench/research_panel_gpqa_diamond_medium.yaml": "专家组 + 证据 + 辩论 + 反思",
    "bench/agentic_repair_swebench_verified_medium.yaml": "定位 + 双补丁 + 测试选择",
}

BENCH_DATASET_NOTES = {
    "bench/chain_gsm8k_medium.yaml": (
        "GSM8K",
        "小学数学文字题，天然适合 parse、solve、check 的短链式逐步推理。",
    ),
    "bench/branch_verify_strategyqa_medium.yaml": (
        "StrategyQA",
        "隐式多步 yes/no 常识题；支持/反驳两条路径能暴露隐藏假设。",
    ),
    "bench/debate_mmlu_pro_medium.yaml": (
        "MMLU-Pro",
        "更难的多任务选择题；多名 debater 并行给出候选答案，再由 judge 裁决。",
    ),
    "bench/self_refine_math_medium.yaml": (
        "MATH",
        "竞赛数学题更容易出现推导漏洞，适合用批改和修正节点提高稳健性。",
    ),
    "bench/plan_code_test_mbpp_medium.yaml": (
        "MBPP",
        "小型 Python 编程题带自然语言需求和测试，适合计划、实现、测试审查。",
    ),
    "bench/research_panel_gpqa_diamond_medium.yaml": (
        "GPQA Diamond",
        "研究生级科学选择题，难度高，适合多专家证据、选项排除、辩论和反思。",
    ),
    "bench/agentic_repair_swebench_verified_medium.yaml": (
        "SWE-bench Verified",
        "真实 GitHub issue 修复任务；需要定位、补丁候选、测试验证和回归风险检查。",
    ),
}


def tex_escape(value: str) -> str:
    table = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(table.get(ch, ch) for ch in str(value))


def collect_edges(ops: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for op_id, spec in ops.items():
        for child in spec.get("output_ops") or []:
            if child in ops:
                edges.add((op_id, child))
        for parent in spec.get("input_ops") or []:
            if parent in ops:
                edges.add((parent, op_id))
    return sorted(edges)


def topo_layers(ops: dict[str, dict[str, Any]], edges: list[tuple[str, str]]) -> dict[str, int]:
    preds = {op_id: set() for op_id in ops}
    succs = {op_id: set() for op_id in ops}
    for src, dst in edges:
        if src in ops and dst in ops:
            preds[dst].add(src)
            succs[src].add(dst)

    indeg = {op_id: len(parents) for op_id, parents in preds.items()}
    queue = deque(sorted(op_id for op_id, deg in indeg.items() if deg == 0))
    layers = {op_id: 0 for op_id in ops}
    seen: list[str] = []
    while queue:
        src = queue.popleft()
        seen.append(src)
        for dst in sorted(succs[src]):
            layers[dst] = max(layers[dst], layers[src] + 1)
            indeg[dst] -= 1
            if indeg[dst] == 0:
                queue.append(dst)

    if len(seen) != len(ops):
        last = max(layers.values(), default=0)
        for op_id in ops:
            if op_id not in seen:
                last += 1
                layers[op_id] = last
    return layers


def load_template(yaml_path: Path) -> tuple[dict[str, dict[str, Any]], set[str], set[str], list[tuple[str, str]]]:
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    ops = data.get("ops") or {}
    if not isinstance(ops, dict) or not ops:
        raise ValueError(f"{yaml_path} has no non-empty ops mapping")

    start_ops = set(data.get("start_ops") or [])
    end_ops = set(data.get("end_ops") or [])
    edges = collect_edges(ops)
    return ops, start_ops, end_ops, edges


def layout_nodes(
    ops: dict[str, dict[str, Any]],
    edges: list[tuple[str, str]],
    *,
    x_gap: float,
    y_gap: float,
    orientation: str = "vertical",
) -> dict[str, tuple[float, float]]:
    layers = topo_layers(ops, edges)
    by_layer: dict[int, list[str]] = defaultdict(list)
    for op_id in ops:
        by_layer[layers[op_id]].append(op_id)
    for layer_ops in by_layer.values():
        layer_ops.sort()

    node_pos: dict[str, tuple[float, float]] = {}
    for layer_idx in sorted(by_layer):
        layer_ops = by_layer[layer_idx]
        span = (len(layer_ops) - 1) * (y_gap if orientation == "horizontal" else x_gap)
        for i, op_id in enumerate(layer_ops):
            if orientation == "horizontal":
                node_pos[op_id] = (layer_idx * x_gap, -(i * y_gap - span / 2))
            else:
                node_pos[op_id] = (i * x_gap - span / 2, -layer_idx * y_gap)
    return node_pos


def layer_stats(ops: dict[str, dict[str, Any]], edges: list[tuple[str, str]]) -> tuple[int, int]:
    layers = topo_layers(ops, edges)
    counts: dict[int, int] = defaultdict(int)
    for layer in layers.values():
        counts[layer] += 1
    return max(layers.values(), default=0) + 1, max(counts.values(), default=1)


def choose_overview_orientation(rel_yaml: str, ops: dict[str, dict[str, Any]], edges: list[tuple[str, str]]) -> str:
    if rel_yaml in {
        "bench/chain_gsm8k_medium.yaml",
        "bench/self_refine_math_medium.yaml",
        "bench/plan_code_test_mbpp_medium.yaml",
    }:
        return "horizontal"
    depth, width = layer_stats(ops, edges)
    return "horizontal" if depth >= 4 and width <= 2 else "vertical"


def render_template(yaml_path: Path, templates_dir: Path, output_dir: Path) -> Path:
    rel_yaml = yaml_path.relative_to(templates_dir)
    ops, start_ops, end_ops, edges = load_template(yaml_path)
    node_pos = layout_nodes(ops, edges, x_gap=3.6, y_gap=1.9)

    tex_path = output_dir / rel_yaml.with_suffix(".tex")
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    layer_widths: dict[float, int] = defaultdict(int)
    for _, y in node_pos.values():
        layer_widths[y] += 1
    widest_layer = max(layer_widths.values(), default=1)
    scale = 0.92 if widest_layer <= 4 else 0.82
    op_to_node = {op_id: f"n{i}" for i, op_id in enumerate(ops)}
    title = tex_escape(rel_yaml.as_posix())

    lines = [
        r"\documentclass[tikz,border=6pt]{standalone}",
        r"\usetikzlibrary{arrows.meta,positioning}",
        r"\begin{document}",
        rf"\begin{{tikzpicture}}[>=Latex, scale={scale}, transform shape,",
        r"  op/.style={circle, draw, align=center, minimum size=11mm, inner sep=0pt, font=\scriptsize, fill=white},",
        r"  start/.style={op, fill=green!12, draw=green!45!black},",
        r"  end/.style={op, fill=red!10, draw=red!45!black},",
        r"  startend/.style={op, fill=yellow!16, draw=orange!55!black},",
        r"  edge/.style={->, line width=0.45pt, draw=black!70}",
        r"]",
        rf"\node[font=\small\bfseries, align=center] at (0,0.95) {{{title}\\[-1pt]{{\scriptsize {len(ops)} ops, {len(edges)} edges}}}};",
    ]
    for op_id in ops:
        x, y = node_pos[op_id]
        if op_id in start_ops and op_id in end_ops:
            style = "startend"
        elif op_id in start_ops:
            style = "start"
        elif op_id in end_ops:
            style = "end"
        else:
            style = "op"
        label = tex_escape(short_node_label(op_id))
        lines.append(rf"\node[{style}] ({op_to_node[op_id]}) at ({x:.2f},{y:.2f}) {{\texttt{{{label}}}}};")

    for src, dst in edges:
        lines.append(rf"\draw[edge] ({op_to_node[src]}) -- ({op_to_node[dst]});")

    if start_ops or end_ops:
        legend_y = min(y for _, y in node_pos.values()) - 0.95
        lines.append(rf"\node[start, minimum width=0.75cm, minimum height=0.45cm, font=\tiny] at (-1.50,{legend_y:.2f}) {{start}};")
        lines.append(rf"\node[end, minimum width=0.75cm, minimum height=0.45cm, font=\tiny] at (1.50,{legend_y:.2f}) {{end}};")

    lines.extend([r"\end{tikzpicture}", r"\end{document}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return tex_path


def render_inline_tikz(yaml_path: Path, templates_dir: Path) -> list[str]:
    rel_yaml = yaml_path.relative_to(templates_dir).as_posix()
    ops, start_ops, end_ops, edges = load_template(yaml_path)
    orientation = choose_overview_orientation(rel_yaml, ops, edges)
    node_pos = layout_nodes(ops, edges, x_gap=1.08, y_gap=0.76, orientation=orientation)
    op_to_node = {op_id: f"n{i}" for i, op_id in enumerate(ops)}
    min_x = min(x for x, _ in node_pos.values())
    max_x = max(x for x, _ in node_pos.values())
    max_y = max(y for _, y in node_pos.values())
    title_x = (min_x + max_x) / 2
    title_y = max_y + 0.55
    lines = [
        r"\begin{tikzpicture}[>=Latex, scale=0.66, transform shape,",
        r"  op/.style={circle, draw, align=center, minimum size=6.7mm, inner sep=0pt, font=\tiny, fill=white},",
        r"  start/.style={op, fill=green!12, draw=green!45!black},",
        r"  end/.style={op, fill=red!10, draw=red!45!black},",
        r"  startend/.style={op, fill=yellow!16, draw=orange!55!black},",
        r"  edge/.style={->, line width=0.35pt, draw=black!65}",
        r"]",
        rf"\node[font=\scriptsize\bfseries, align=center] at ({title_x:.2f},{title_y:.2f}) {{{tex_escape(Path(rel_yaml).name)}\\[-1pt]{{\tiny {len(ops)} ops, {len(edges)} edges}}}};",
    ]
    for op_id in ops:
        x, y = node_pos[op_id]
        if op_id in start_ops and op_id in end_ops:
            style = "startend"
        elif op_id in start_ops:
            style = "start"
        elif op_id in end_ops:
            style = "end"
        else:
            style = "op"
        label = tex_escape(short_node_label(op_id))
        lines.append(rf"\node[{style}] ({op_to_node[op_id]}) at ({x:.2f},{y:.2f}) {{{label}}};")
    for src, dst in edges:
        lines.append(rf"\draw[edge] ({op_to_node[src]}) -- ({op_to_node[dst]});")
    lines.append(r"\end{tikzpicture}")
    return lines


def short_node_label(op_id: str) -> str:
    words = [part for part in op_id.replace("-", "_").split("_") if part]
    if not words:
        return op_id[:3]
    if len(words) == 1:
        return words[0][:3]
    return "".join(word[0] for word in words[:3])


def render_overview_cell(yaml_path: Path, templates_dir: Path, width_cm: float) -> str:
    rel_yaml = yaml_path.relative_to(templates_dir).as_posix()
    dag_type = BENCH_DAG_TYPES.get(rel_yaml, "通用 DAG")
    dataset, note = BENCH_DATASET_NOTES.get(
        rel_yaml,
        ("待选数据集", "通用 DAG 形状，用于测试调度行为。"),
    )
    if rel_yaml in BENCH_COMPLEX:
        frame = "red!72!black"
        fill = "red!2"
    else:
        frame = "blue!65!black"
        fill = "blue!2"
    body_lines = [
        r"\centering",
        *render_inline_tikz(yaml_path, templates_dir),
        r"\\[-2pt]",
        rf"{{\scriptsize\bfseries 形态：{tex_escape(dag_type)}}}\\",
        rf"{{\scriptsize\bfseries 数据集：{tex_escape(dataset)}}}\\[-1pt]",
        rf"{{\scriptsize {tex_escape(note)}}}",
    ]
    inner_width = width_cm - 0.35
    return "\n".join(
        [
            rf"\fcolorbox{{{frame}}}{{{fill}}}{{\begin{{minipage}}[t]{{{inner_width:.2f}cm}}",
            *body_lines,
            r"\end{minipage}}",
        ]
    )


def render_bench_overview(templates_dir: Path, output_dir: Path) -> Path:
    basic_paths = [templates_dir / rel for rel in BENCH_BASIC_ORDER if (templates_dir / rel).is_file()]
    complex_paths = [templates_dir / rel for rel in BENCH_COMPLEX_ORDER if (templates_dir / rel).is_file()]
    tex_path = output_dir / "bench" / "bench_overview.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    basic_cells = [render_overview_cell(path, templates_dir, 7.16) for path in basic_paths]
    complex_cells = [render_overview_cell(path, templates_dir, 17.95) for path in complex_paths]
    while len(basic_cells) < 5:
        basic_cells.append("")
    while len(complex_cells) < 2:
        complex_cells.append("")

    lines = [
        r"\documentclass[border=8pt]{standalone}",
        r"\usepackage[UTF8]{ctex}",
        r"\usepackage{tikz}",
        r"\usetikzlibrary{arrows.meta}",
        r"\usepackage{array}",
        r"\usepackage{xcolor}",
        r"\begin{document}",
        r"\begin{minipage}{37.8cm}",
        r"\begin{center}",
        r"{\Large\bfseries Bench DAG 设计总览：先定形态，再配数据集}\\[3pt]",
        r"{\small 圆形节点表示 MFE operator；绿色为起点，红色为终点；蓝框为基础形态，红框为复杂综合形态。}\\[8pt]",
        r"{\normalsize\bfseries 基础形态（由简单到复杂）}\\[4pt]",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.18}",
        r"\begin{tabular}{*{5}{>{\centering\arraybackslash}p{7.16cm}}}",
        "\n&\n".join(basic_cells[:5]) + r"\\[12pt]",
        r"\end{tabular}",
        r"\\[2pt]",
        r"{\normalsize\bfseries 复杂综合形态（尽量覆盖上面的结构特征）}\\[4pt]",
        r"\begin{tabular}{*{2}{>{\centering\arraybackslash}p{17.95cm}}}",
        "\n&\n".join(complex_cells[:2]) + r"\\",
        r"\end{tabular}",
        r"\end{center}",
        r"\end{minipage}",
        r"\end{document}",
        "",
    ]
    tex_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return tex_path


def write_index(output_dir: Path, templates_dir: Path, tex_paths: list[Path], overview_path: Path | None = None) -> None:
    visible_tex_paths = tex_paths
    if overview_path is not None:
        visible_rel = set(BENCH_ORDER)
        visible_tex_paths = [
            tex_path
            for tex_path in tex_paths
            if tex_path.relative_to(output_dir).with_suffix(".yaml").as_posix() in visible_rel
        ]
    rows = [
        "# Template DAG diagrams",
        "",
        "Generated from `templates/**/*.yaml`.",
        "",
        "The combined bench overview uses the redesigned DAG set documented in `bench/dag_dataset_selection.md`.",
        "",
        "| YAML | TikZ source | PDF |",
        "| --- | --- | --- |",
    ]
    if overview_path is not None:
        rel_tex = overview_path.relative_to(output_dir).as_posix()
        rel_pdf = overview_path.relative_to(output_dir).with_suffix(".pdf").as_posix()
        rows.append(f"| `templates/bench/*.yaml overview` | `{rel_tex}` | `{rel_pdf}` |")
        rel_png = overview_path.relative_to(output_dir).with_suffix(".png").as_posix()
        rows.append(f"| `templates/bench/*.yaml overview PNG` | `{rel_png}` | `{rel_png}` |")
    for tex_path in sorted(visible_tex_paths):
        rel_tex = tex_path.relative_to(output_dir).as_posix()
        rel_yaml = tex_path.relative_to(output_dir).with_suffix(".yaml").as_posix()
        rel_pdf = tex_path.relative_to(output_dir).with_suffix(".pdf").as_posix()
        rows.append(f"| `templates/{rel_yaml}` | `{rel_tex}` | `{rel_pdf}` |")
    rows.extend([
        "",
        "Regenerate diagrams with:",
        "",
        "```bash",
        "python -m mfe.scripts.generate_template_dags --bench-only --bench-overview --compile-pdf --clean",
        "```",
        "",
    ])
    (output_dir / "README.md").write_text("\n".join(rows), encoding="utf-8", newline="\n")


def compile_pdf(tex_path: Path, *, engine: str = "pdf") -> None:
    engine_flag = "-xelatex" if engine == "xelatex" else "-pdf"
    subprocess.run(
        ["latexmk", "-cd", engine_flag, "-interaction=nonstopmode", "-halt-on-error", "-quiet", str(tex_path)],
        check=True,
    )


def clean_latex_aux(output_dir: Path) -> None:
    aux_suffixes = {".aux", ".log", ".fls", ".fdb_latexmk", ".xdv", ".synctex.gz"}
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if path.suffix in aux_suffixes or name.endswith(".synctex.gz"):
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TikZ DAG diagrams for YAML templates")
    parser.add_argument("--templates-dir", default="templates")
    parser.add_argument("--output-dir", default="docs/template-dags")
    parser.add_argument("--bench-only", action="store_true", help="generate only templates/bench/*.yaml")
    parser.add_argument("--bench-overview", action="store_true", help="also generate a combined bench overview figure")
    parser.add_argument("--compile-pdf", action="store_true")
    parser.add_argument("--clean", action="store_true", help="remove LaTeX auxiliary files after compilation")
    args = parser.parse_args()

    templates_dir = Path(args.templates_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if args.bench_only:
        paths = [templates_dir / rel for rel in BENCH_ORDER if (templates_dir / rel).is_file()]
    else:
        paths = sorted(templates_dir.rglob("*.yaml"))
    tex_paths = [render_template(path, templates_dir, output_dir) for path in paths]
    overview_path = render_bench_overview(templates_dir, output_dir) if args.bench_overview else None
    write_index(output_dir, templates_dir, tex_paths, overview_path)
    if args.compile_pdf:
        for tex_path in tex_paths:
            compile_pdf(tex_path)
        if overview_path is not None:
            compile_pdf(overview_path, engine="xelatex")
    if args.clean:
        clean_latex_aux(output_dir)
    suffix = " and bench overview" if overview_path is not None else ""
    print(f"generated {len(tex_paths)} DAG diagrams{suffix} under {output_dir}")


if __name__ == "__main__":
    main()
