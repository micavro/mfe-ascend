import json
import tempfile
import unittest
from pathlib import Path

from mfe.scripts.summarize_scheduler_runs import (
    collect_metrics,
    compact_lines,
    markdown_table,
    write_outputs,
)


class SchedulerRunSummaryTest(unittest.TestCase):
    def _write_run(self, root: Path, scheduler: str, offset: float) -> None:
        run_dir = root / scheduler
        run_dir.mkdir()
        details = [
            {
                "status": "completed",
                "arrive_time": 10.0,
                "done_time": 20.0 + offset,
                "idle_time": 2.0,
                "service_time": 8.0 + offset,
                "latency": 10.0 + offset,
            },
            {
                "status": "completed",
                "arrive_time": 14.0,
                "done_time": 25.0 + offset,
                "idle_time": 3.0,
                "service_time": 8.0 + offset,
                "latency": 11.0 + offset,
            },
        ]
        summary = {
            "scheduler": scheduler,
            "count": 2,
            "completed": 2,
            "success_rate": 1.0,
            "makespan": 15.0 + offset,
            "total_token_throughput": 123.0 + offset,
            "ready_queue_avg": 1.5 + offset,
            "ready_queue_peak": 4,
            "device_busy_pct": {"0": 0.5, "1": 0.7},
        }
        (run_dir / f"fixture_{scheduler}.json").write_text(json.dumps(details), encoding="utf-8")
        (run_dir / f"fixture_{scheduler}_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        (run_dir / "summary_all.json").write_text(json.dumps([summary]), encoding="utf-8")

    def test_collects_required_metrics_and_writes_brief(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, scheduler in enumerate(("fcfs", "sjf", "rhsail")):
                self._write_run(root, scheduler, float(index))

            rows = collect_metrics(root, ["fcfs", "sjf", "rhsail"], expected_count=2)
            self.assertEqual(3, len(rows))
            self.assertAlmostEqual(4.0, rows[0].arrival_end_s)
            self.assertAlmostEqual(11.0, rows[0].drain_tail_s)
            self.assertAlmostEqual(2.5, rows[0].avg_wait_s)
            self.assertAlmostEqual(8.0, rows[0].avg_service_s)
            self.assertAlmostEqual(10.5, rows[0].avg_completion_s)
            self.assertAlmostEqual(60.0, rows[0].device_busy_pct)

            table = markdown_table(rows)
            self.assertIn("RH-SAIL", table)
            self.assertIn("Ready avg/peak", table)
            self.assertIn("device_busy=60.0%", compact_lines(rows))
            write_outputs(root, rows)
            self.assertTrue((root / "final_brief.md").is_file())
            self.assertTrue((root / "final_brief.csv").is_file())
            self.assertIn("MFE_FINAL_BRIEF_START", (root / "final_brief.txt").read_text())

    def test_rejects_incomplete_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_run(root, "fcfs", 0.0)
            summary_path = next((root / "fcfs").glob("*_summary.json"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["completed"] = 1
            summary["success_rate"] = 0.5
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete run"):
                collect_metrics(root, ["fcfs"], expected_count=2)


if __name__ == "__main__":
    unittest.main()
