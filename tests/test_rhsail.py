import unittest

from mfe.components import Operator, Query
from mfe.optimizers.rhsail import RHSailReadyScheduler


def build_chain():
    root = Operator(id="root")
    middle = Operator(id="middle")
    final = Operator(id="final")
    root.output_ops = [middle]
    middle.input_ops = [root]
    middle.output_ops = [final]
    final.input_ops = [middle]
    return {"root": root, "middle": middle, "final": final}, [root], [final]


class RHSailSchedulerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ops, self.start_ops, self.end_ops = build_chain()

        def get_dag(_template):
            return self.ops, self.start_ops, self.end_ops

        self.get_dag = get_dag

    def make_query(self, uid: str, create_time: float) -> Query:
        query = Query(
            uid,
            "question",
            template="chain.yaml",
            metadata={"input_len_est_tokens": 128},
        )
        query.create_time = create_time
        return query

    def scheduler(self, workers: int = 1) -> RHSailReadyScheduler:
        scheduler = RHSailReadyScheduler(workers, use_sail_guidance=False)
        scheduler.hard_stretch = 1000.0
        scheduler.soft_stretch = 1000.0
        return scheduler

    def test_active_limit_suppresses_new_start_when_active_work_is_available(self) -> None:
        scheduler = self.scheduler()
        scheduler.active_dag_limit = 1
        scheduler.admission_max_wait = 1000.0

        active = self.make_query("active", 0.0)
        active.status = "running"
        active.op_output = {"root": "done"}
        active.step = 1
        active.benchmark = {"root": [10.0, 11.0]}

        waiting = self.make_query("waiting", 90.0)
        requests = {"active": active, "waiting": waiting}
        ready = [("waiting", self.ops["root"]), ("active", self.ops["middle"])]

        ordered = scheduler.order_ready_tasks(
            ready,
            worker_id=0,
            requests=requests,
            inflight_tasks=set(),
            get_dag=self.get_dag,
            now=100.0,
        )

        self.assertEqual(("active", "middle"), (ordered[0][0], ordered[0][1].id))

    def test_hard_inter_op_gap_preempts_recent_active_workflow(self) -> None:
        scheduler = self.scheduler()
        scheduler.soft_gap_seconds = 5.0
        scheduler.hard_gap_seconds = 10.0
        scheduler.commitment_grace_seconds = 1000.0

        old = self.make_query("old", 0.0)
        old.status = "running"
        old.op_output = {"root": "done"}
        old.step = 2
        old.benchmark = {"root": [0.0, 1.0]}

        recent = self.make_query("recent", 90.0)
        recent.status = "running"
        recent.op_output = {"root": "done"}
        recent.step = 2
        recent.benchmark = {"root": [95.0, 96.0]}

        ordered = scheduler.order_ready_tasks(
            [("recent", self.ops["middle"]), ("old", self.ops["middle"])],
            worker_id=0,
            requests={"old": old, "recent": recent},
            inflight_tasks=set(),
            get_dag=self.get_dag,
            now=100.0,
        )

        self.assertEqual("old", ordered[0][0])

    def test_initial_progress_commitment_prevents_one_op_abandonment(self) -> None:
        scheduler = self.scheduler()
        scheduler.soft_gap_seconds = 30.0
        scheduler.hard_gap_seconds = 180.0
        scheduler.commitment_grace_seconds = 30.0
        scheduler.commitment_ops = 2

        committed = self.make_query("committed", 0.0)
        committed.status = "running"
        committed.op_output = {"root": "done"}
        committed.step = 1
        committed.benchmark = {"root": [49.0, 50.0]}

        progressed = self.make_query("progressed", 0.0)
        progressed.status = "running"
        progressed.op_output = {"root": "done"}
        progressed.step = 2
        progressed.benchmark = {"root": [74.0, 75.0]}

        ordered = scheduler.order_ready_tasks(
            [("progressed", self.ops["middle"]), ("committed", self.ops["middle"])],
            worker_id=0,
            requests={"committed": committed, "progressed": progressed},
            inflight_tasks=set(),
            get_dag=self.get_dag,
            now=81.0,
        )

        self.assertEqual("committed", ordered[0][0])

    def test_parent_worker_affinity_is_used_as_a_soft_tie_break(self) -> None:
        scheduler = self.scheduler(workers=2)
        scheduler.w_cp = 0.0
        scheduler.w_unlock = 0.0
        scheduler.w_finish = 0.0
        scheduler.w_gap = 0.0
        scheduler.w_stretch = 0.0
        scheduler.w_commitment = 0.0
        scheduler.w_admission = 0.0
        scheduler.w_short = 0.0
        scheduler.w_affinity = 1.0
        scheduler.eta_affinity = 1.0

        local = self.make_query("local", 80.0)
        local.status = "running"
        local.op_output = {"root": "done"}
        local.step = 2
        local.benchmark = {"root": [90.0, 91.0]}
        local.worker_assignments = {"root": 1}

        remote = self.make_query("remote", 80.0)
        remote.status = "running"
        remote.op_output = {"root": "done"}
        remote.step = 2
        remote.benchmark = {"root": [90.0, 91.0]}
        remote.worker_assignments = {"root": 0}

        ordered = scheduler.order_ready_tasks(
            [("remote", self.ops["middle"]), ("local", self.ops["middle"])],
            worker_id=1,
            requests={"local": local, "remote": remote},
            inflight_tasks=set(),
            get_dag=self.get_dag,
            now=92.0,
        )

        self.assertEqual("local", ordered[0][0])

    def test_finishing_operator_reduces_rollout_completion_cost(self) -> None:
        scheduler = self.scheduler()
        scheduler.soft_gap_seconds = 1000.0
        scheduler.hard_gap_seconds = 2000.0

        finishing = self.make_query("finishing", 0.0)
        finishing.status = "running"
        finishing.op_output = {"root": "done", "middle": "done"}
        finishing.step = 2
        finishing.benchmark = {"root": [90.0, 91.0], "middle": [92.0, 93.0]}

        waiting = self.make_query("waiting", 99.0)
        ordered = scheduler.order_ready_tasks(
            [("waiting", self.ops["root"]), ("finishing", self.ops["final"])],
            worker_id=0,
            requests={"finishing": finishing, "waiting": waiting},
            inflight_tasks=set(),
            get_dag=self.get_dag,
            now=100.0,
        )

        self.assertEqual(("finishing", "final"), (ordered[0][0], ordered[0][1].id))

    def test_runtime_observation_replaces_uniform_max_token_estimate(self) -> None:
        scheduler = self.scheduler()
        query = self.make_query("q", 0.0)

        scheduler.observe_completion(
            query,
            "root",
            worker_id=0,
            benchmark=[0.0, 9.0],
            metrics={"input_tokens": 130, "output_tokens": 20},
        )

        self.assertAlmostEqual(9.0, scheduler._estimate_runtime(query, self.ops["root"]))
        self.assertEqual(1, scheduler.snapshot()["runtime_model_buckets"])


if __name__ == "__main__":
    unittest.main()
