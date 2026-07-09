import os
import tempfile
import unittest

from mfe.components import ExecuteInfo, Operator, Query
from mfe.optimizers.multi_request import MultiRequestOptimizer
from mfe.scripts.client import _extract_final_answer
from mfe.workers.worker_test import TestWorker


class PromptContractTest(unittest.TestCase):
    def test_build_prompt_uses_only_direct_parent_generated_outputs(self) -> None:
        optimizer = object.__new__(MultiRequestOptimizer)
        query = Query("q1", "ROOT QUESTION", template="bench/example.yaml")
        query.op_output = {
            "ancestor": "ANCESTOR OUTPUT SHOULD NOT BE INCLUDED",
            "parent": "PARENT GENERATED OUTPUT",
        }
        optimizer.requests = {"q1": query}

        parent = Operator(id="parent")
        child = Operator(id="child")
        child.input_ops = [parent]

        prompt = optimizer._build_prompt("q1", child)

        self.assertIn("ROOT QUESTION", prompt)
        self.assertIn("[parent output]\nPARENT GENERATED OUTPUT", prompt)
        self.assertNotIn("ANCESTOR OUTPUT SHOULD NOT BE INCLUDED", prompt)

    def test_resolve_template_path_prefers_templates_dir_for_relative_paths(self) -> None:
        optimizer = object.__new__(MultiRequestOptimizer)
        with tempfile.TemporaryDirectory() as tmpdir:
            bench_dir = os.path.join(tmpdir, "bench")
            os.makedirs(bench_dir)
            expected = os.path.join(bench_dir, "example.yaml")
            with open(expected, "w", encoding="utf-8") as f:
                f.write("start_ops: []\n")
            optimizer.templates_dir = tmpdir

            self.assertEqual(os.path.abspath(expected), optimizer._resolve_template_path("bench/example.yaml"))

    def test_extract_final_answer_returns_latest_generated_output(self) -> None:
        status = {
            "op_output": {"first": "first generated", "final": "final generated"},
            "benchmark": {"first": [0.0, 1.0], "final": [1.0, 2.0]},
        }

        self.assertEqual("final generated", _extract_final_answer(status))

    def test_test_worker_output_does_not_echo_prompt(self) -> None:
        old_delay = os.environ.get("MFE_TEST_WORKER_DELAY")
        old_output = os.environ.get("MFE_TEST_OUTPUT_TEXT")
        os.environ["MFE_TEST_WORKER_DELAY"] = "0"
        os.environ["MFE_TEST_OUTPUT_TEXT"] = "generated only"
        try:
            worker = TestWorker(0, 0, None, None)
            op = Operator(id="op1")
            result = worker.execute(ExecuteInfo(op=op, query_ids=["q1"], prompts=["ROOT QUESTION"]))
        finally:
            if old_delay is None:
                os.environ.pop("MFE_TEST_WORKER_DELAY", None)
            else:
                os.environ["MFE_TEST_WORKER_DELAY"] = old_delay
            if old_output is None:
                os.environ.pop("MFE_TEST_OUTPUT_TEXT", None)
            else:
                os.environ["MFE_TEST_OUTPUT_TEXT"] = old_output

        self.assertEqual("generated only", result["item"][0]["output"])
        self.assertNotIn("ROOT QUESTION", result["item"][0]["output"])


if __name__ == "__main__":
    unittest.main()
