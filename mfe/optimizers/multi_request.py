"""多请求异步优化器。

默认行为保持原来的 eager ready-task 调度；设置 MFE_SCHEDULER=sailp 或
MFE_ENABLE_SAILP=1 后，提交请求时会运行 SAI-LP admission-time planner，
调度循环按计划 worker/timeline 派发 ready op。
"""

from __future__ import annotations

from logging import getLogger
import math
import multiprocessing as mp
import os
import queue
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from mfe.components import ExecuteInfo, Operator, Query
from mfe.config import is_verbose
from mfe.parser import build_ops_from_config, load_config
from mfe.util import (
    get_accelerator_backend,
    no_visible_devices_message,
    visible_accelerator_device_ids,
)
from mfe.workers import TestWorker, vLLMWorker
from mfe.optimizers.sailp import SAILPScheduler, SchedulePlan

logger = getLogger(__name__)
logger.setLevel("INFO")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _worker_process(
    worker_id: int,
    physical_device_id: int,
    cmd_queue: mp.Queue,
    result_queue: mp.Queue,
    use_test_worker: bool = False,
) -> None:
    try:
        (TestWorker if use_test_worker else vLLMWorker)(
            worker_id,
            physical_device_id,
            cmd_queue,
            result_queue,
        ).run()
    except Exception as exc:
        result_queue.put(
            {
                "command": "error",
                "result": {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "op_name": "?",
                    "worker_id": worker_id,
                },
                "elapsed_time": 0.0,
            }
        )


class MultiRequestOptimizer:
    def __init__(
        self,
        templates_dir: str = "templates",
        use_test_worker: bool | None = None,
    ) -> None:
        self.templates_dir = os.path.abspath(templates_dir)
        self._template_cache: Dict[str, Tuple[Dict[str, Operator], List[Operator], List[Operator], Any]] = {}
        self._lock = threading.RLock()
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        if use_test_worker is None:
            use_test_worker = os.environ.get("MFE_USE_TEST_WORKER", "").lower() in (
                "1",
                "true",
                "yes",
            )
        self._use_test_worker = use_test_worker
        self._backend = get_accelerator_backend()
        if self._use_test_worker:
            self.device_cnt = int(os.environ.get("MFE_TEST_DEVICE_CNT", "4"))
            device_ids = list(range(self.device_cnt))
        else:
            device_ids = visible_accelerator_device_ids(self._backend)
            self.device_cnt = len(device_ids)
        if not device_ids:
            raise RuntimeError(no_visible_devices_message(self._backend))

        scheduler_name = os.environ.get("MFE_SCHEDULER", os.environ.get("MFE_OPTIMIZER", "eager"))
        scheduler_name = scheduler_name.strip().lower()
        self._scheduler_name = scheduler_name
        self._sailp_enabled = scheduler_name in {"sail", "sailp", "sai-lp", "state", "state-aware"} or _env_bool(
            "MFE_ENABLE_SAILP", False
        )
        self._sailp_strict = _env_bool("MFE_SAILP_STRICT", False)
        self._sailp_planner: Optional[SAILPScheduler] = None
        self._scheduler_overhead_seconds = 0.0
        self._scheduler_decisions = 0
        self._ready_queue_samples: List[int] = []
        if self._sailp_enabled:
            self._sailp_planner = SAILPScheduler(num_executors=self.device_cnt, executor_ids=list(range(self.device_cnt)))

        self.cmd_queues: List[mp.Queue] = []
        self.result_queues: List[mp.Queue] = []
        for _ in range(self.device_cnt):
            self.cmd_queues.append(mp.Queue())
            self.result_queues.append(mp.Queue())

        self.processes: List[mp.Process] = []
        for i, device_id in enumerate(device_ids):
            proc = mp.Process(
                target=_worker_process,
                args=(i, device_id, self.cmd_queues[i], self.result_queues[i], self._use_test_worker),
                daemon=False,
            )
            self.processes.append(proc)
            proc.start()

        self.requests: Dict[str, Query] = {}
        self._inflight: List[Optional[Tuple[str, Operator]]] = [None] * self.device_cnt
        self._inflight_tasks: Set[Tuple[str, str]] = set()
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        logger.info(
            "MultiRequestOptimizer initialized (templates_dir=%s, backend=%s, devices=%d, use_test_worker=%s, scheduler=%s)",
            self.templates_dir,
            self._backend,
            self.device_cnt,
            self._use_test_worker,
            "sailp" if self._sailp_enabled else "eager",
        )

    def _resolve_template_path(self, template: str) -> str:
        t = (template or "").strip()
        if not t:
            raise ValueError("template is empty")
        if os.path.isabs(t):
            candidates = [t]
        else:
            candidates = [os.path.join(self.templates_dir, t), os.path.abspath(t)]
        seen: List[str] = []
        for candidate in candidates:
            path = os.path.abspath(candidate)
            if path in seen:
                continue
            seen.append(path)
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"template not found: {t}; tried: {', '.join(seen)}")

    def _get_dag(self, template: str) -> Tuple[Dict[str, Operator], List[Operator], List[Operator]]:
        path = self._resolve_template_path(template)
        model_override = os.environ.get("MFE_MODEL_PATH") or None
        cache_key = f"{path}::{model_override or ''}"
        if cache_key in self._template_cache:
            ops, start_ops, end_ops, _ = self._template_cache[cache_key]
            return ops, start_ops, end_ops
        config = load_config(path)
        ops, start_ops, end_ops, models = build_ops_from_config(config, model_override=model_override)
        self._template_cache[cache_key] = (ops, start_ops, end_ops, models)
        return ops, start_ops, end_ops

    def _ensure_schedule_plan(self, q: Query) -> Optional[SchedulePlan]:
        if not self._sailp_enabled or self._sailp_planner is None:
            q.scheduler_name = "eager"
            return None
        if q.schedule_plan is not None:
            return q.schedule_plan
        ops, _, _ = self._get_dag(q.template)
        plan = self._sailp_planner.plan(ops)
        q.schedule_plan = plan
        q.scheduler_name = "sailp"
        if is_verbose():
            print(
                f"[OPT] SAI-LP plan query={q.id[:8]} template={q.template} "
                f"makespan={plan.makespan:.3f} method={plan.guidance_method}",
                flush=True,
            )
            for wid, timeline in plan.timelines.items():
                print(f"[OPT]   worker {wid}: {timeline}", flush=True)
        return plan

    def submit(self, dag: str, input_text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        uid = str(uuid.uuid4())
        tpl = dag if dag.endswith(".yaml") else f"{dag}.yaml"
        q = Query(id=uid, prompt=input_text or "", template=tpl, metadata=metadata)
        q.scheduler_name = "sailp" if self._sailp_enabled else self._scheduler_name
        with self._lock:
            if self._sailp_enabled:
                self._ensure_schedule_plan(q)
            self.requests[uid] = q
        return uid

    def status(self, uid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            q = self.requests.get(uid)
            if q is None:
                return None
            if q.status == "error":
                return {
                    "uid": uid,
                    "status": "error",
                    "op_output": dict(q.op_output),
                    "benchmark": {k: [float(t[0]), float(t[1])] for k, t in q.benchmark.items()},
                    "worker_assignments": dict(q.worker_assignments),
                    "op_metrics": dict(q.op_metrics),
                    "schedule_plan": q.schedule_plan.to_dict() if q.schedule_plan is not None else None,
                    "scheduler": getattr(q, "scheduler_name", "eager"),
                    "scheduler_metrics": self._scheduler_snapshot(),
                    "total_answer_time": None,
                    "arrive_time": q.create_time,
                    "done_time": None,
                    "error_type": q.error_type,
                    "error_message": q.error_message,
                    "failed_op": q.failed_op,
                    "worker_id": q.worker_id,
                }
            try:
                ops, _, end_ops = self._get_dag(q.template)
                if self._sailp_enabled:
                    self._ensure_schedule_plan(q)
            except Exception:
                return {
                    "uid": uid,
                    "status": "error",
                    "op_output": {},
                    "benchmark": {},
                    "worker_assignments": dict(q.worker_assignments),
                    "op_metrics": dict(q.op_metrics),
                    "schedule_plan": None,
                    "scheduler": "sailp" if self._sailp_enabled else self._scheduler_name,
                    "scheduler_metrics": self._scheduler_snapshot(),
                    "total_answer_time": None,
                    "arrive_time": q.create_time,
                    "done_time": None,
                    "error_type": "TemplateError",
                    "error_message": f"failed to load template: {q.template}",
                    "failed_op": None,
                    "worker_id": None,
                }
            end_ids = {e.id for e in end_ops}
            done = set(q.op_output.keys())
            if end_ids and end_ids <= done:
                st = "completed"
            elif done or any(uid == x[0] for x in self._inflight if x is not None):
                st = "running"
            else:
                st = "pending"
            done_time = max(t[1] for t in q.benchmark.values()) if q.benchmark else None
            return {
                "uid": uid,
                "status": st,
                "op_output": dict(q.op_output),
                "benchmark": {k: [float(t[0]), float(t[1])] for k, t in q.benchmark.items()},
                "worker_assignments": dict(q.worker_assignments),
                "op_metrics": dict(q.op_metrics),
                "schedule_plan": q.schedule_plan.to_dict() if q.schedule_plan is not None else None,
                "scheduler": getattr(q, "scheduler_name", "eager"),
                "scheduler_metrics": self._scheduler_snapshot(),
                "total_answer_time": (done_time - q.create_time if done_time is not None else None),
                "arrive_time": q.create_time,
                "done_time": done_time,
            }

    def _scheduler_snapshot(self) -> Dict[str, Any]:
        samples = list(self._ready_queue_samples)
        return {
            "scheduler": "sailp" if self._sailp_enabled else self._scheduler_name,
            "overhead_seconds": float(self._scheduler_overhead_seconds),
            "decisions": int(self._scheduler_decisions),
            "ready_queue_samples": len(samples),
            "ready_queue_avg": (sum(samples) / len(samples) if samples else 0.0),
            "ready_queue_peak": (max(samples) if samples else 0),
        }

    def _is_query_complete(self, q: Query) -> bool:
        try:
            _, _, end_ops = self._get_dag(q.template)
        except Exception:
            return False
        end_ids = {e.id for e in end_ops}
        return bool(end_ids) and end_ids <= set(q.op_output.keys())

    def _ready_ops_for_query(self, uid: str, q: Query) -> List[Operator]:
        try:
            ops, _, _ = self._get_dag(q.template)
        except Exception:
            return []
        ready: List[Operator] = []
        for op in ops.values():
            if op.id in q.op_output or (uid, op.id) in self._inflight_tasks:
                continue
            if all(p.id in q.op_output for p in op.input_ops):
                ready.append(op)
        return ready

    def _get_ready_tasks(self) -> List[Tuple[str, Operator]]:
        ready: List[Tuple[str, Operator]] = []
        with self._lock:
            for uid, q in list(self.requests.items()):
                if q.status == "error" or self._is_query_complete(q):
                    continue
                for op in self._ready_ops_for_query(uid, q):
                    ready.append((uid, op))
        return self._order_ready_tasks(ready)

    def _order_ready_tasks(self, ready: List[Tuple[str, Operator]]) -> List[Tuple[str, Operator]]:
        if self._sailp_enabled:
            return ready
        if self._scheduler_name in {"sjf", "shortest-job-first", "short-job-first"}:
            with self._lock:
                return sorted(
                    ready,
                    key=lambda item: (
                        int(getattr(self.requests.get(item[0]), "input_len_est_tokens", 0) or 0),
                        float(getattr(self.requests.get(item[0]), "create_time", 0.0) or 0.0),
                        item[0],
                        item[1].id,
                    ),
                )
        return ready

    def _get_ready_tasks_for_worker(self, worker_id: int) -> List[Tuple[str, Operator]]:
        """Return ready tasks ordered by SAI-LP plan for a specific idle worker."""
        if not self._sailp_enabled:
            return self._get_ready_tasks()
        strict_candidates: List[Tuple[Tuple[float, float, int], str, Operator]] = []
        fallback_candidates: List[Tuple[Tuple[float, float, int], str, Operator]] = []
        with self._lock:
            for uid, q in list(self.requests.items()):
                if q.status == "error" or self._is_query_complete(q):
                    continue
                plan = self._ensure_schedule_plan(q)
                for op in self._ready_ops_for_query(uid, q):
                    if plan is None:
                        fallback_candidates.append(((math.inf, math.inf, 10**9), uid, op))
                        continue
                    pri = plan.priority_for(op.id)
                    planned_worker = plan.worker_for(op.id)
                    if planned_worker == worker_id:
                        strict_candidates.append((pri, uid, op))
                    else:
                        fallback_candidates.append((pri, uid, op))
        strict_candidates.sort(key=lambda x: x[0])
        if strict_candidates:
            return [(uid, op) for _, uid, op in strict_candidates]
        if self._sailp_strict:
            return []
        fallback_candidates.sort(key=lambda x: x[0])
        return [(uid, op) for _, uid, op in fallback_candidates]

    @staticmethod
    def _ordinal_label(index: int) -> str:
        n = index + 1
        suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    def _prompt_context_ops(self, op: Operator) -> List[Operator]:
        ordered: List[Operator] = []
        visited: Set[str] = set()
        visiting: Set[str] = set()

        def visit(node: Operator) -> None:
            node_id = str(node.id)
            if node_id in visited or node_id in visiting:
                return
            visiting.add(node_id)
            for parent in node.input_ops:
                visit(parent)
            visiting.remove(node_id)
            visited.add(node_id)
            ordered.append(node)

        for parent in op.input_ops:
            visit(parent)
        return ordered

    def _prompt_context_sections(self, op: Operator) -> Tuple[List[Operator], List[Tuple[Operator, List[Operator]]]]:
        parents = list(op.input_ops)
        if len(parents) <= 1:
            return ([], [(parents[0], self._prompt_context_ops(op))] if parents else [])

        branch_paths: List[Tuple[Operator, List[Operator]]] = []
        branch_sets: List[Set[str]] = []
        for parent in parents:
            path = self._prompt_context_ops(parent)
            path.append(parent)
            branch_paths.append((parent, path))
            branch_sets.append({str(node.id) for node in path})

        common_ids = set.intersection(*branch_sets) if branch_sets else set()
        common_ops = [node for node in branch_paths[0][1] if str(node.id) in common_ids]
        emitted = set(common_ids)
        branches: List[Tuple[Operator, List[Operator]]] = []
        for parent, path in branch_paths:
            branch_ops: List[Operator] = []
            for node in path:
                node_id = str(node.id)
                if node_id in emitted:
                    continue
                emitted.add(node_id)
                branch_ops.append(node)
            branches.append((parent, branch_ops))
        return common_ops, branches

    def _build_prompt(self, uid: str, op: Operator) -> str:
        q = self.requests[uid]
        parts: List[str] = []
        if q.prompt:
            parts.append(q.prompt.strip())

        common_ops, branches = self._prompt_context_sections(op)
        if common_ops and len(op.input_ops) > 1:
            common_parts: List[str] = []
            for ctx_op in common_ops:
                output = q.op_output.get(ctx_op.id, "")
                if output:
                    common_parts.append(f"[{ctx_op.id} output]\n{output.strip()}")
            if common_parts:
                parts.append("[shared upstream context]\n" + "\n\n".join(common_parts))

        for branch_index, (parent, branch_ops) in enumerate(branches):
            branch_parts: List[str] = []
            for ctx_op in branch_ops:
                output = q.op_output.get(ctx_op.id, "")
                if output:
                    branch_parts.append(f"[{ctx_op.id} output]\n{output.strip()}")
            if not branch_parts:
                continue
            if len(op.input_ops) > 1:
                parts.append(
                    f"[{self._ordinal_label(branch_index)} branch via {parent.id}]\n"
                    + "\n\n".join(branch_parts)
                )
                continue
            parts.extend(branch_parts)
        return "\n\n".join(parts)

    def _handle_worker_result(self, worker_id: int, msg: Any, uid: str, op: Operator) -> None:
        if isinstance(msg, dict) and msg.get("command") == "execute":
            result = msg.get("result", {})
            op_name = result.get("op_name") or result.get("node_name")
            if is_verbose() and op_name and uid:
                print(
                    f"[OPT] <- Worker {worker_id} op_name={op_name} query_ids=[{uid}] "
                    f"t={time.perf_counter():.1f}",
                    flush=True,
                )
            if op_name and uid:
                with self._lock:
                    for rec in result.get("item", []):
                        q = self.requests.get(rec.get("id"))
                        if q:
                            q.op_output[op_name] = rec["output"]
                            q.step += 1
                            q.status = "running"
                            q.benchmark[op_name] = rec["benchmark"]
                            if isinstance(rec.get("metrics"), dict):
                                q.op_metrics[op_name] = rec["metrics"]
        elif isinstance(msg, dict) and msg.get("command") == "error":
            err = msg.get("result", {})
            if not isinstance(err, dict):
                err = {"error_type": "WorkerError", "error_message": str(err)}
            with self._lock:
                q = self.requests.get(uid)
                if q:
                    q.status = "error"
                    q.error_type = err.get("error_type", "WorkerError")
                    q.error_message = err.get("error_message", "worker failed")
                    q.failed_op = err.get("op_name") or getattr(op, "id", None)
                    q.worker_id = err.get("worker_id", worker_id)
                    q.worker_assignments[getattr(op, "id", "?")] = worker_id
            if is_verbose():
                print(
                    f"[OPT] !! Worker {worker_id} op={getattr(op, 'id', '?')} "
                    f"error={err.get('error_message')}",
                    flush=True,
                )

    def _dispatch(self, worker_id: int, uid: str, op: Operator) -> None:
        prompt = self._build_prompt(uid, op)
        exe = ExecuteInfo(op=op, query_ids=[uid], prompts=[prompt])
        if is_verbose() and not op.input_ops:
            ops_dict, _, _ = self._get_dag(self.requests[uid].template)
            print(
                f"[OPT] query id={uid[:8]} template={self.requests[uid].template} "
                f"DAG_ops={list(ops_dict.keys())}",
                flush=True,
            )
        self.cmd_queues[worker_id].put(("execute", (exe,)))
        self._inflight[worker_id] = (uid, op)
        self._inflight_tasks.add((uid, op.id))
        with self._lock:
            q = self.requests.get(uid)
            if q:
                q.status = "running"
                q.worker_assignments[op.id] = worker_id
        if is_verbose():
            plan_note = ""
            q = self.requests.get(uid)
            if q and q.schedule_plan is not None:
                step = q.schedule_plan.steps.get(op.id)
                if step is not None:
                    plan_note = f" planned_worker={step.worker_id} reuse={step.reuse_from or 'cold'}"
            print(
                f"[OPT] -> Worker {worker_id} op={op.id} query_ids=[{uid}]{plan_note} "
                f"t={time.perf_counter():.1f}",
                flush=True,
            )

    def _scheduler_loop(self) -> None:
        while self._running:
            for i in range(self.device_cnt):
                if self._inflight[i] is None:
                    continue
                try:
                    msg = self.result_queues[i].get(timeout=0.1)
                except queue.Empty:
                    continue
                uid, op = self._inflight[i]
                self._inflight[i] = None
                if op:
                    self._inflight_tasks.discard((uid, op.id))
                self._handle_worker_result(i, msg, uid, op)

            for i in range(self.device_cnt):
                if self._inflight[i] is not None:
                    continue
                sched_start = time.perf_counter()
                ready = self._get_ready_tasks_for_worker(i)
                self._scheduler_overhead_seconds += time.perf_counter() - sched_start
                self._scheduler_decisions += 1
                self._ready_queue_samples.append(len(ready))
                if not ready:
                    continue
                uid, op = ready[0]
                self._dispatch(i, uid, op)
            time.sleep(0.01)

    def exit(self) -> None:
        self._running = False
        if self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=2.0)
        for i in range(self.device_cnt):
            self.cmd_queues[i].put(("exit", ()))
        for q in self.cmd_queues + self.result_queues:
            q.close()
            q.join_thread()
        for p in self.processes:
            p.join()
        logger.info("MultiRequestOptimizer exited")
