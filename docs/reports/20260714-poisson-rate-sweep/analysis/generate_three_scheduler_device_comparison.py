#!/usr/bin/env python3
"""Draw a shared-axis 0.12/0.15 device-occupancy comparison."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def find_workspace_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "remote_runs").is_dir() and (candidate / "mfe-ascend").is_dir():
            return candidate
    raise RuntimeError("Could not locate workspace root")


ROOT = find_workspace_root()
RUN_012 = ROOT / "remote_runs" / (
    "20260713-234500-full-first200-poisson012-013-fcfs-sjf-rhsail-"
    "vllm5-max32768-out2048-mem075"
)
RUN_015_BASE = ROOT / "remote_runs" / (
    "20260711-024102-full-first200-poisson015-batch1-fcfs-sjf-sailp-"
    "vllm5-max32768-out2048-mem075"
)
RUN_015_RHSAIL = ROOT / "remote_runs" / (
    "20260713-160000-full-first200-poisson015-rhsail-darc-"
    "vllm5-max32768-out2048-mem075"
)
PUBLISH_DIR = (
    ROOT
    / "mfe-ascend"
    / "docs"
    / "reports"
    / "20260714-poisson-rate-sweep"
    / "figures"
)
REMOTE_FIGURE_DIR = RUN_012 / "figures"
OUTPUT_NAME = "device_occupancy_rate012_rate015_three_schedulers.png"

SCHEDULERS = ["fcfs", "sjf", "rhsail"]
LABELS = {"fcfs": "FCFS", "sjf": "SJF", "rhsail": "RH-SAIL"}
RATES = ["0.12", "0.15"]
PHYSICAL_GPUS = {str(index): index + 3 for index in range(5)}
X_MAX_MINUTES = 205.0

WIDTH = 3000
HEIGHT = 1960
OUTER_X = 60
COLUMN_GAP = 60
COLUMN_WIDTH = (WIDTH - 2 * OUTER_X - COLUMN_GAP) // 2
TOP = 175
PANEL_HEIGHT = 510
ROW_GAP = 25
PLOT_LEFT_OFFSET = 160
PLOT_RIGHT_PAD = 28
LANE_TOP_OFFSET = 62
LANE_HEIGHT = 38
LANE_STEP = 58

BACKGROUND = "#ffffff"
FOREGROUND = "#17212b"
MUTED = "#58636e"
GRID = "#d8dee4"
LANE_BACKGROUND = "#f4f6f8"
RUNNING = "#2f8f5b"
ARRIVAL = "#4f5963"
LAST_ARRIVAL = "#a33a3a"
RATE_012 = "#4078b7"
RATE_015 = "#b4662f"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


FONT_TITLE = font(42, bold=True)
FONT_SUBTITLE = font(29, bold=True)
FONT_PANEL = font(23, bold=True)
FONT_LABEL = font(20)
FONT_TICK = font(17)
FONT_LEGEND = font(19)


def only_match(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern} under {root}, found {matches}")
    return matches[0]


def result_root(rate: str, scheduler: str) -> Path:
    if rate == "0.12":
        return RUN_012 / "rate012" / scheduler
    if scheduler == "rhsail":
        return RUN_015_RHSAIL / scheduler
    return RUN_015_BASE / scheduler


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
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


def load_panel(rate: str, scheduler: str) -> dict[str, Any]:
    root = result_root(rate, scheduler)
    rows = json.loads(only_match(root, "*_run1.json").read_text(encoding="utf-8"))
    summary = json.loads(
        only_match(root, "*_run1_summary.json").read_text(encoding="utf-8")
    )
    if len(rows) != 1400 or int(summary["completed"]) != 1400:
        raise RuntimeError(f"Incomplete data for {rate}/{scheduler}")

    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        assignments = row.get("worker_assignments") or {}
        for op, span in (row.get("benchmark") or {}).items():
            if op not in assignments or not isinstance(span, list) or len(span) < 2:
                continue
            start, end = float(span[0]), float(span[1])
            if end > start:
                intervals[str(assignments[op])].append((start, end))

    arrivals = sorted(float(row["arrive_time"]) for row in rows)
    makespan = float(summary["makespan"])
    if makespan / 60.0 > X_MAX_MINUTES:
        raise RuntimeError(f"Shared x-axis too short for {rate}/{scheduler}: {makespan}")
    return {
        "arrivals": arrivals,
        "arrival_end": max(arrivals),
        "makespan": makespan,
        "busy_pct": {
            worker: float(summary["device_busy_pct"][worker]) * 100.0
            for worker in PHYSICAL_GPUS
        },
        "intervals": {
            worker: merge_intervals(intervals[worker]) for worker in PHYSICAL_GPUS
        },
    }


def x_coordinate(seconds: float, left: int, right: int) -> int:
    fraction = max(0.0, min(seconds / (X_MAX_MINUTES * 60.0), 1.0))
    return int(round(left + fraction * (right - left)))


def dashed_vertical(
    draw: ImageDraw.ImageDraw, x: int, top: int, bottom: int, color: str
) -> None:
    y = top
    while y < bottom:
        draw.line((x, y, x, min(y + 12, bottom)), fill=color, width=2)
        y += 20


def draw_panel(
    draw: ImageDraw.ImageDraw,
    panel: dict[str, Any],
    rate: str,
    scheduler: str,
    column: int,
    row: int,
) -> None:
    x0 = OUTER_X + column * (COLUMN_WIDTH + COLUMN_GAP)
    y0 = TOP + row * (PANEL_HEIGHT + ROW_GAP)
    plot_left = x0 + PLOT_LEFT_OFFSET
    plot_right = x0 + COLUMN_WIDTH - PLOT_RIGHT_PAD
    lanes_top = y0 + LANE_TOP_OFFSET
    lanes_bottom = lanes_top + 5 * LANE_STEP + LANE_HEIGHT

    drain = max(0.0, panel["makespan"] - panel["arrival_end"])
    header = (
        f"{LABELS[scheduler]}  |  makespan {panel['makespan'] / 60:.1f} min"
        f"  |  drain {drain / 60:.1f} min"
    )
    draw.text((x0 + 4, y0 + 5), header, font=FONT_PANEL, fill=FOREGROUND)

    for tick in range(0, 201, 25):
        x = x_coordinate(tick * 60.0, plot_left, plot_right)
        draw.line((x, lanes_top, x, lanes_bottom), fill=GRID, width=1)
        if row == len(SCHEDULERS) - 1:
            draw.text(
                (x, lanes_bottom + 12),
                str(tick),
                font=FONT_TICK,
                fill=MUTED,
                anchor="ma",
            )

    labels = ["Query arrivals"] + [f"GPU {gpu}" for gpu in range(3, 8)]
    for lane_index, label in enumerate(labels):
        lane_y = lanes_top + lane_index * LANE_STEP
        draw.rectangle(
            (plot_left, lane_y, plot_right, lane_y + LANE_HEIGHT),
            fill=LANE_BACKGROUND,
            outline=GRID,
            width=1,
        )
        if lane_index == 0:
            suffix = ""
        else:
            worker = str(lane_index - 1)
            suffix = f"  {panel['busy_pct'][worker]:.1f}%"
        draw.text(
            (plot_left - 12, lane_y + LANE_HEIGHT / 2),
            label + suffix,
            font=FONT_LABEL,
            fill=FOREGROUND,
            anchor="rm",
        )

    arrival_top = lanes_top + 5
    arrival_bottom = lanes_top + LANE_HEIGHT - 5
    for arrival in panel["arrivals"]:
        x = x_coordinate(arrival, plot_left, plot_right)
        draw.line((x, arrival_top, x, arrival_bottom), fill=ARRIVAL, width=1)

    for worker, intervals in panel["intervals"].items():
        lane_index = int(worker) + 1
        lane_y = lanes_top + lane_index * LANE_STEP
        for start, end in intervals:
            start_x = x_coordinate(start, plot_left, plot_right)
            end_x = max(start_x + 1, x_coordinate(end, plot_left, plot_right))
            draw.rectangle(
                (start_x, lane_y + 3, end_x, lane_y + LANE_HEIGHT - 3),
                fill=RUNNING,
            )

    arrival_end_x = x_coordinate(panel["arrival_end"], plot_left, plot_right)
    dashed_vertical(draw, arrival_end_x, lanes_top, lanes_bottom, LAST_ARRIVAL)

    if row == len(SCHEDULERS) - 1:
        draw.text(
            ((plot_left + plot_right) / 2, lanes_bottom + 48),
            f"Elapsed time (minutes), shared scale 0–{int(X_MAX_MINUTES)}",
            font=FONT_LABEL,
            fill=FOREGROUND,
            anchor="ma",
        )


def main() -> None:
    panels = {
        (rate, scheduler): load_panel(rate, scheduler)
        for rate in RATES
        for scheduler in SCHEDULERS
    }
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.text(
        (WIDTH / 2, 38),
        "Device occupancy: Poisson 0.12 vs 0.15 (same 0–205 min axis)",
        font=FONT_TITLE,
        fill=FOREGROUND,
        anchor="ma",
    )
    column_titles = [
        ("0.12 req/s — near capacity, short drain", RATE_012),
        ("0.15 req/s — overloaded, persistent drain", RATE_015),
    ]
    for column, (title, color) in enumerate(column_titles):
        center = OUTER_X + column * (COLUMN_WIDTH + COLUMN_GAP) + COLUMN_WIDTH / 2
        draw.text((center, 102), title, font=FONT_SUBTITLE, fill=color, anchor="ma")

    legend_y = 143
    legend_x = WIDTH // 2 - 360
    draw.line((legend_x, legend_y - 10, legend_x, legend_y + 10), fill=ARRIVAL, width=2)
    draw.text((legend_x + 14, legend_y), "Query arrival", font=FONT_LEGEND, fill=FOREGROUND, anchor="lm")
    legend_x += 230
    draw.rectangle((legend_x, legend_y - 10, legend_x + 28, legend_y + 10), fill=RUNNING)
    draw.text((legend_x + 42, legend_y), "DAG op running", font=FONT_LEGEND, fill=FOREGROUND, anchor="lm")
    legend_x += 280
    dashed_vertical(draw, legend_x, legend_y - 13, legend_y + 14, LAST_ARRIVAL)
    draw.text((legend_x + 14, legend_y), "Actual last arrival", font=FONT_LEGEND, fill=FOREGROUND, anchor="lm")

    for column, rate in enumerate(RATES):
        for row, scheduler in enumerate(SCHEDULERS):
            draw_panel(draw, panels[(rate, scheduler)], rate, scheduler, column, row)

    PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
    REMOTE_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    publish_path = PUBLISH_DIR / OUTPUT_NAME
    image.save(publish_path, format="PNG", optimize=True)
    shutil.copy2(publish_path, REMOTE_FIGURE_DIR / OUTPUT_NAME)
    print(f"figure={publish_path}")
    print(f"size={image.width}x{image.height}")
    print(f"shared_x_axis_minutes={X_MAX_MINUTES}")


if __name__ == "__main__":
    main()
