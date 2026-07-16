import csv
import json
import tempfile
import unittest
from pathlib import Path

from mfe.scripts.summarize_scheduler_runs_detailed import (
    build_markdown,
    collect_metrics,
    compact_lines,
    percentile,
    request_max_gap,
    write_outputs,
)


class DetailedSchedulerRunSummaryTest(unittest.TestCase):
    def _write_run(self, root: Path, scheduler: str, offset: float) -> None:
        run_dir = root / scheduler
        run_dir.mkdir()
        details = [
            {
                "status": "completed",
                "dataset": "alpha",
                "arrive_time": 10.0,
                "done_time": 30.0 + offset,
                "idle_time": 5.0,
                "service_time": 15.0 + offset,
                "run_time": 10.0 + offset,
                "latency": 20.0 + offset,
                "benchmark": {"a": [15.0, 20.0], "b": [23.0, 28.0 + offset]},
            },
            {
                "status": "completed",
                "dataset": "beta",
                "arrive_time": 12.0,
                "done_time": 32.0 + offset,
                "idle_time": 4.0,
                "service_time": 16.0 + offset,
                "run_time": 11.0 + offset,
                "latency": 20.0 + offset,
                "benchmark": {"a": [16.0, 21.0], "b": [25.0, 31.0 + offset]},
            },
        ]
        summary = {
            "scheduler": scheduler,
            "count": 2,
            "completed": 2,
            "success_rate": 1.0,
            "makespan": 22.0 + offset,
            "input_tokens": 200,
            "output_tokens": 100,
            "input_token_throughput": 20.0 + offset,
            "output_token_throughput": 10.0 + offset,
            "total_token_throughput": 30.0 + offset,
            "scheduler_overhead_seconds": 2.0 + offset,
            "scheduler_overhead_pct": 0.1,
            "ready_queue_avg": 1.5,
            "ready_queue_peak": 4,
            "device_busy_pct": {"0": 0.5, "1": 0.7},
        }
        (run_dir / f"fixture_{scheduler}.json").write_text(
            json.dumps(details), encoding="utf-8"
        )
        (run_dir / f"fixture_{scheduler}_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

    def test_collects_requested_overall_and_dataset_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            schedulers = ["fcfs", "sjf", "rhsail"]
            for index, scheduler in enumerate(schedulers):
                self._write_run(root, scheduler, float(index))

            overall, datasets = collect_metrics(root, schedulers, expected_count=2)
            self.assertEqual(3, len(overall))
            self.assertEqual(6, len(datasets))
            self.assertAlmostEqual(20.0, overall[0].input_tokens_per_s)
            self.assertAlmostEqual(10.5, overall[0].avg_run_time_s)
            self.assertAlmostEqual(15.99, overall[0].p99_service_s)
            self.assertAlmostEqual(16.0, overall[0].max_service_s)
            self.assertAlmostEqual(20.0, overall[0].p99_completion_s)
            self.assertAlmostEqual(3.95, overall[0].p95_max_gap_s)
            self.assertAlmostEqual(10.0, overall[0].scheduler_overhead_pct)
            self.assertAlmostEqual(60.0, overall[0].device_busy_pct)

            markdown = build_markdown(overall, datasets)
            self.assertIn("Input tok/s", markdown)
            self.assertIn("P99 service", markdown)
            self.assertIn("FCFS service/run(s)", markdown)
            text = compact_lines(overall, datasets)
            self.assertIn("DATASET alpha avg_service/run", text)
            self.assertIn("ready=1.5/4", text)
            write_outputs(root, overall, datasets, "detailed_brief")
            self.assertTrue((root / "detailed_brief.md").is_file())
            self.assertTrue((root / "detailed_brief_overall.csv").is_file())
            with (root / "detailed_brief_by_dataset.csv").open(encoding="utf-8") as handle:
                self.assertEqual(6, len(list(csv.DictReader(handle))))

    def test_percentile_and_overlapping_interval_gap(self) -> None:
        self.assertAlmostEqual(3.7, percentile([1, 2, 3, 4], 90))
        row = {"benchmark": {"a": [0, 3], "b": [2, 5], "c": [9, 10]}}
        self.assertAlmostEqual(4.0, request_max_gap(row))

    def test_rejects_incomplete_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_run(root, "fcfs", 0.0)
            path = next((root / "fcfs").glob("*_summary.json"))
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["completed"] = 1
            path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete run"):
                collect_metrics(root, ["fcfs"], expected_count=2)


if __name__ == "__main__":
    unittest.main()
