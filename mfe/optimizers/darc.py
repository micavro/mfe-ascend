"""DARC scheduler for MFE multi-request optimizer.

DARC = Dependency-Aware Rollout Criticality Scheduler.

This scheduler is intentionally KV-cache agnostic:
- no prefix-cache locality
- no local/remote KV state
- no reuse mode
- no state transfer delay

It is designed for the existing MFE online scheduling loop.  At every idle
worker, MultiRequestOptimizer asks DARC to order currently ready (query, op)
tasks.  DARC ranks tasks with:
- DAG critical-path rank
- downstream unlock value
- request/op aging
- SJF-style shortness
- active-DAG admission throttling
- lightweight limited-horizon rollout

The public entry point is DARCReadyScheduler.order_ready_tasks(...).
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
import math
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from mfe.components import Operator, Query

logger = getLogger(__name__)

ReadyTask = Tuple[str, Operator]
OpsGetter = Callable[[str], Tuple[Dict[str, Operator], List[Operator], List[Operator]]]

_EPS = 1e-9


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid float env %s=%r", name, raw)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring invalid int env %s=%r", name, raw)
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class DARCTaskScore:
    uid: str
    op_id: str
    score: float
    rollout_cost: float
    critical_path: float
    unlock_work: float
    age: float
    runtime: float
    is_start_op: bool


class DARCReadyScheduler:
    """Online ready-task scheduler for MultiRequestOptimizer.

    This class does not dispatch by itself.  It only orders the ready task list
    returned by MultiRequestOptimizer._ready_ops_for_query().
    """

    def __init__(self, num_workers: int) -> None:
        self.num_workers = max(1, int(num_workers))

        # Active DAG control.
        # If MFE_DARC_ACTIVE_DAG_LIMIT <= 0, use factor * num_workers.
        raw_limit = _env_int("MFE_DARC_ACTIVE_DAG_LIMIT", 0)
        self.active_dag_limit = raw_limit if raw_limit > 0 else int(
            math.ceil(_env_float("MFE_DARC_ACTIVE_DAG_FACTOR", 3.0) * self.num_workers)
        )
        self.active_dag_limit = max(1, self.active_dag_limit)

        # Candidate + rollout.
        self.candidate_k = max(1, _env_int("MFE_DARC_CANDIDATE_K", 8))
        self.rollout_horizon = max(1, _env_int("MFE_DARC_ROLLOUT_HORIZON", 3))

        # Priority weights.
        self.w_cp = _env_float("MFE_DARC_W_CP", 0.35)
        self.w_unlock = _env_float("MFE_DARC_W_UNLOCK", 0.25)
        self.w_age = _env_float("MFE_DARC_W_AGE", 0.20)
        self.w_short = _env_float("MFE_DARC_W_SHORT", 0.15)
        self.w_stall = _env_float("MFE_DARC_W_STALL", 0.05)

        # Rollout weights.
        self.eta_flow = _env_float("MFE_DARC_ETA_FLOW", 1.0)
        self.eta_cp = _env_float("MFE_DARC_ETA_CP", 0.5)
        self.eta_ready = _env_float("MFE_DARC_ETA_READY", 0.2)

        # Runtime estimator.  No KV-cache terms.
        self.default_op_time = _env_float("MFE_DARC_DEFAULT_OP_TIME", 1.0)
        self.input_token_time = _env_float("MFE_DARC_INPUT_TOKEN_TIME", 0.001)
        self.output_token_time = _env_float("MFE_DARC_OUTPUT_TOKEN_TIME", 0.003)
        self.min_runtime = _env_float("MFE_DARC_MIN_RUNTIME", 1e-6)

        # Aging.
        self.aging_cap_seconds = max(1.0, _env_float("MFE_DARC_AGING_CAP_SECONDS", 600.0))
        self.use_request_age = _env_bool("MFE_DARC_USE_REQUEST_AGE", True)

        # Whether to log score detail.
        self.verbose = _env_bool("MFE_DARC_VERBOSE", False)

        # Per-template DAG metric cache.
        # key = template string.
        self._metric_cache: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def order_ready_tasks(
        self,
        ready: List[ReadyTask],
        *,
        worker_id: int,
        requests: Mapping[str, Query],
        inflight_tasks: Set[Tuple[str, str]],
        get_dag: OpsGetter,
        now: Optional[float] = None,
    ) -> List[ReadyTask]:
        """Return ready tasks ordered by DARC.

        Args:
            ready: list of (uid, Operator)
            worker_id: currently idle worker id
            requests: MultiRequestOptimizer.requests
            inflight_tasks: MultiRequestOptimizer._inflight_tasks
            get_dag: function template -> (ops, start_ops, end_ops)
            now: current perf_counter time

        Returns:
            ordered ready list. MultiRequestOptimizer dispatches ready[0].
        """
        if not ready:
            return []

        now = time.perf_counter() if now is None else float(now)

        # 1. Active-DAG throttling.
        filtered = self._apply_active_limit(
            ready,
            requests=requests,
            inflight_tasks=inflight_tasks,
            get_dag=get_dag,
        )
        if not filtered:
            filtered = ready

        # 2. Score candidates.
        scored: List[Tuple[float, ReadyTask, DARCTaskScore]] = []
        for task in filtered:
            uid, op = task
            q = requests.get(uid)
            if q is None:
                continue
            score_obj = self._score_task(
                uid,
                op,
                q,
                requests=requests,
                inflight_tasks=inflight_tasks,
                get_dag=get_dag,
                now=now,
            )
            scored.append((score_obj.score, task, score_obj))

        if not scored:
            return ready

        scored.sort(key=lambda x: (-x[0], self._query_create_time(requests.get(x[1][0])), x[1][0], x[1][1].id))
        candidates = scored[: self.candidate_k]

        # 3. Limited-horizon rollout over top-K.
        final: List[Tuple[float, float, ReadyTask, DARCTaskScore]] = []
        for _, task, score_obj in candidates:
            cost = self._rollout_cost(
                first_task=task,
                ready=filtered,
                requests=requests,
                inflight_tasks=inflight_tasks,
                get_dag=get_dag,
                now=now,
            )
            # Lower cost is better.  Tie-break by higher score.
            final.append((cost, -score_obj.score, task, score_obj))

        final.sort(key=lambda x: (x[0], x[1], self._query_create_time(requests.get(x[2][0])), x[2][0], x[2][1].id))

        best_tasks = [x[2] for x in final]
        best_set = {(uid, op.id) for uid, op in best_tasks}

        # Append non-candidate tasks by original score, so caller still sees all options.
        rest = [
            task
            for _, task, _ in scored
            if (task[0], task[1].id) not in best_set
        ]
        ordered = best_tasks + rest

        if self.verbose and ordered:
            best = final[0][3]
            print(
                "[DARC] worker={} choose uid={} op={} score={:.4f} rollout={:.4f} "
                "cp={:.3f} unlock={:.3f} age={:.3f} runtime={:.3f} active_limit={}".format(
                    worker_id,
                    best.uid[:8],
                    best.op_id,
                    best.score,
                    final[0][0],
                    best.critical_path,
                    best.unlock_work,
                    best.age,
                    best.runtime,
                    self.active_dag_limit,
                ),
                flush=True,
            )

        return ordered

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": "darc",
            "active_dag_limit": self.active_dag_limit,
            "candidate_k": self.candidate_k,
            "rollout_horizon": self.rollout_horizon,
            "weights": {
                "critical_path": self.w_cp,
                "unlock": self.w_unlock,
                "age": self.w_age,
                "short": self.w_short,
                "stall": self.w_stall,
            },
            "rollout_weights": {
                "flow": self.eta_flow,
                "cp": self.eta_cp,
                "ready": self.eta_ready,
            },
        }

    # ------------------------------------------------------------------
    # Active DAG admission
    # ------------------------------------------------------------------

    def _apply_active_limit(
        self,
        ready: List[ReadyTask],
        *,
        requests: Mapping[str, Query],
        inflight_tasks: Set[Tuple[str, str]],
        get_dag: OpsGetter,
    ) -> List[ReadyTask]:
        active = self._active_query_ids(requests, inflight_tasks, get_dag)
        active_count = len(active)

        # If active count is below limit, no throttling.
        if active_count < self.active_dag_limit:
            return ready

        # If saturated, suppress start ops from not-yet-active queries.
        kept: List[ReadyTask] = []
        for uid, op in ready:
            if uid in active:
                kept.append((uid, op))
                continue
            q = requests.get(uid)
            if q is None:
                continue
            if not self._is_start_op(q, op, get_dag):
                kept.append((uid, op))

        # If everything is suppressed, allow the oldest start request to avoid deadlock.
        if kept:
            return kept

        return sorted(
            ready,
            key=lambda item: (
                self._query_create_time(requests.get(item[0])),
                item[0],
                item[1].id,
            ),
        )[:1]

    def _active_query_ids(
        self,
        requests: Mapping[str, Query],
        inflight_tasks: Set[Tuple[str, str]],
        get_dag: OpsGetter,
    ) -> Set[str]:
        active: Set[str] = set()
        inflight_uids = {uid for uid, _ in inflight_tasks}
        for uid, q in requests.items():
            if getattr(q, "status", None) == "error":
                continue
            if uid in inflight_uids:
                active.add(uid)
                continue
            if getattr(q, "op_output", None):
                if not self._query_complete(q, get_dag):
                    active.add(uid)
        return active

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_task(
        self,
        uid: str,
        op: Operator,
        q: Query,
        *,
        requests: Mapping[str, Query],
        inflight_tasks: Set[Tuple[str, str]],
        get_dag: OpsGetter,
        now: float,
    ) -> DARCTaskScore:
        metrics = self._template_metrics(q.template, get_dag)
        op_id = str(op.id)

        runtime = self._estimate_runtime(q, op)
        cp = float(metrics["cp"].get(op_id, runtime))
        unlock = float(metrics["unlock"].get(op_id, 0.0))
        template_cp = max(_EPS, float(metrics["template_cp"]))

        cp_norm = cp / template_cp
        unlock_norm = unlock / template_cp

        if self.use_request_age:
            age = max(0.0, now - self._query_create_time(q))
        else:
            # There is no per-op ready timestamp in current MFE, so fallback to query age.
            age = max(0.0, now - self._query_create_time(q))
        age_norm = min(1.0, age / self.aging_cap_seconds)

        short = 1.0 / max(runtime, self.min_runtime)
        short_norm = short / (1.0 + short)

        stall = self._internal_stall_risk(q, op, get_dag, inflight_tasks)

        score = (
            self.w_cp * cp_norm
            + self.w_unlock * unlock_norm
            + self.w_age * age_norm
            + self.w_short * short_norm
            - self.w_stall * stall
        )

        return DARCTaskScore(
            uid=uid,
            op_id=op_id,
            score=score,
            rollout_cost=0.0,
            critical_path=cp,
            unlock_work=unlock,
            age=age,
            runtime=runtime,
            is_start_op=self._is_start_op(q, op, get_dag),
        )

    def _estimate_runtime(self, q: Query, op: Operator) -> float:
        """Estimate op runtime without KV-cache effects."""
        spec = _safe_mapping(getattr(op, "sailp", None))

        for key in ("cold_time", "estimated_time", "duration", "cost"):
            val = _as_float(spec.get(key), None)
            if val is not None:
                return max(self.min_runtime, val)

        cfg = getattr(op, "model_config", None)
        output_tokens = _as_float(getattr(cfg, "max_tokens", None), None)
        if output_tokens is None:
            output_tokens = _as_float(os.environ.get("MFE_OUTPUT_MAX_TOKENS"), 256.0) or 256.0

        input_tokens = _as_float(getattr(q, "input_len_est_tokens", None), 0.0) or 0.0

        # For non-start ops, upstream outputs are part of prompt.  We only use
        # a conservative output-token count approximation; no KV reuse.
        done_outputs = len(getattr(q, "op_output", {}) or {})
        input_tokens += done_outputs * max(1.0, output_tokens * 0.25)

        runtime = (
            self.default_op_time
            + self.input_token_time * input_tokens
            + self.output_token_time * output_tokens
        )
        return max(self.min_runtime, runtime)

    def _internal_stall_risk(
        self,
        q: Query,
        op: Operator,
        get_dag: OpsGetter,
        inflight_tasks: Set[Tuple[str, str]],
    ) -> float:
        try:
            ops, _, _ = get_dag(q.template)
        except Exception:
            return 0.0

        done = set(getattr(q, "op_output", {}) or {})
        inflight = {op_id for uid, op_id in inflight_tasks if uid == q.id}
        unfinished = [oid for oid in ops if oid not in done and oid not in inflight]
        if not unfinished:
            return 0.0

        ready_count = 0
        for oid in unfinished:
            node = ops[oid]
            if all(parent.id in done for parent in getattr(node, "input_ops", []) or []):
                ready_count += 1

        return max(0.0, (len(unfinished) - ready_count) / max(1, len(unfinished)))

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _rollout_cost(
        self,
        *,
        first_task: ReadyTask,
        ready: List[ReadyTask],
        requests: Mapping[str, Query],
        inflight_tasks: Set[Tuple[str, str]],
        get_dag: OpsGetter,
        now: float,
    ) -> float:
        """Lightweight limited-horizon rollout.

        This is a shadow simulation over DAG readiness.  It does not execute
        model calls and does not model KV cache.
        """
        shadow_done: Dict[str, Set[str]] = {
            uid: set(getattr(q, "op_output", {}) or {})
            for uid, q in requests.items()
        }
        shadow_inflight: Set[Tuple[str, str]] = set(inflight_tasks)
        shadow_ready: List[ReadyTask] = list(ready)

        total = 0.0
        t = now
        chosen: Optional[ReadyTask] = first_task

        for h in range(self.rollout_horizon):
            if chosen is None:
                break

            uid, op = chosen
            q = requests.get(uid)
            if q is None:
                break

            op_id = str(op.id)
            runtime = self._estimate_runtime(q, op)

            # Execute chosen op in shadow.
            shadow_ready = [
                item for item in shadow_ready
                if not (item[0] == uid and item[1].id == op_id)
            ]
            shadow_inflight.discard((uid, op_id))
            shadow_done.setdefault(uid, set()).add(op_id)

            # Unlock successors.
            try:
                ops, _, _ = get_dag(q.template)
                for child in getattr(op, "output_ops", []) or []:
                    cid = str(child.id)
                    if cid not in ops:
                        continue
                    if cid in shadow_done[uid] or (uid, cid) in shadow_inflight:
                        continue
                    child_op = ops[cid]
                    if all(parent.id in shadow_done[uid] for parent in getattr(child_op, "input_ops", []) or []):
                        if not any(xuid == uid and xop.id == cid for xuid, xop in shadow_ready):
                            shadow_ready.append((uid, child_op))
            except Exception:
                pass

            unfinished = self._shadow_unfinished_count(requests, shadow_done, get_dag)
            remaining_cp = self._shadow_remaining_cp(requests, shadow_done, get_dag)

            total += (
                self.eta_flow * unfinished * runtime
                + self.eta_cp * remaining_cp
                + self.eta_ready * len(shadow_ready)
            )

            t += runtime

            if h + 1 >= self.rollout_horizon or not shadow_ready:
                break

            # Greedy continuation in shadow.
            chosen = self._shadow_best_task(
                shadow_ready,
                requests=requests,
                shadow_done=shadow_done,
                get_dag=get_dag,
                now=t,
            )

        return total

    def _shadow_best_task(
        self,
        shadow_ready: List[ReadyTask],
        *,
        requests: Mapping[str, Query],
        shadow_done: Mapping[str, Set[str]],
        get_dag: OpsGetter,
        now: float,
    ) -> Optional[ReadyTask]:
        best: Optional[ReadyTask] = None
        best_score = -math.inf

        for uid, op in shadow_ready:
            q = requests.get(uid)
            if q is None:
                continue
            metrics = self._template_metrics(q.template, get_dag)
            op_id = str(op.id)
            runtime = self._estimate_runtime(q, op)
            cp = float(metrics["cp"].get(op_id, runtime))
            unlock = float(metrics["unlock"].get(op_id, 0.0))
            template_cp = max(_EPS, float(metrics["template_cp"]))

            age = max(0.0, now - self._query_create_time(q))
            age_norm = min(1.0, age / self.aging_cap_seconds)
            short = 1.0 / max(runtime, self.min_runtime)
            short_norm = short / (1.0 + short)

            score = (
                self.w_cp * (cp / template_cp)
                + self.w_unlock * (unlock / template_cp)
                + self.w_age * age_norm
                + self.w_short * short_norm
            )
            if score > best_score:
                best_score = score
                best = (uid, op)

        return best

    def _shadow_unfinished_count(
        self,
        requests: Mapping[str, Query],
        shadow_done: Mapping[str, Set[str]],
        get_dag: OpsGetter,
    ) -> int:
        total = 0
        for uid, q in requests.items():
            try:
                _, _, end_ops = get_dag(q.template)
            except Exception:
                continue
            done = shadow_done.get(uid, set())
            end_ids = {str(op.id) for op in end_ops}
            if not end_ids or not end_ids <= done:
                total += 1
        return total

    def _shadow_remaining_cp(
        self,
        requests: Mapping[str, Query],
        shadow_done: Mapping[str, Set[str]],
        get_dag: OpsGetter,
    ) -> float:
        total = 0.0
        for uid, q in requests.items():
            metrics = self._template_metrics(q.template, get_dag)
            done = shadow_done.get(uid, set())
            remain = [
                cp
                for op_id, cp in metrics["cp"].items()
                if op_id not in done
            ]
            if remain:
                total += max(remain)
        return total

    # ------------------------------------------------------------------
    # DAG metrics
    # ------------------------------------------------------------------

    def _template_metrics(self, template: str, get_dag: OpsGetter) -> Dict[str, Any]:
        key = str(template)
        cached = self._metric_cache.get(key)
        if cached is not None:
            return cached

        ops, start_ops, _ = get_dag(template)

        cp: Dict[str, float] = {}
        visiting: Set[str] = set()

        def dfs(op: Operator) -> float:
            oid = str(op.id)
            if oid in cp:
                return cp[oid]
            if oid in visiting:
                # Parser should already reject cycles.  Be defensive.
                return self.default_op_time
            visiting.add(oid)

            own = self._estimate_operator_static_runtime(op)
            children = [c for c in getattr(op, "output_ops", []) or [] if str(c.id) in ops]
            if not children:
                cp[oid] = own
            else:
                cp[oid] = own + max(dfs(child) for child in children)
            visiting.remove(oid)
            return cp[oid]

        for op in ops.values():
            dfs(op)

        unlock: Dict[str, float] = {}
        for oid, op in ops.items():
            val = 0.0
            for child in getattr(op, "output_ops", []) or []:
                cid = str(child.id)
                val += cp.get(cid, self.default_op_time)
            unlock[oid] = val

        start_ids = [str(op.id) for op in start_ops]
        template_cp = max((cp.get(oid, 0.0) for oid in start_ids), default=max(cp.values(), default=1.0))
        total_work = sum(self._estimate_operator_static_runtime(op) for op in ops.values())
        parallelism = total_work / max(template_cp, _EPS)

        result = {
            "cp": cp,
            "unlock": unlock,
            "template_cp": max(_EPS, template_cp),
            "total_work": total_work,
            "parallelism": parallelism,
            "start_ids": set(start_ids),
        }
        self._metric_cache[key] = result
        return result

    def _estimate_operator_static_runtime(self, op: Operator) -> float:
        spec = _safe_mapping(getattr(op, "sailp", None))
        for key in ("cold_time", "estimated_time", "duration", "cost"):
            val = _as_float(spec.get(key), None)
            if val is not None:
                return max(self.min_runtime, val)

        cfg = getattr(op, "model_config", None)
        output_tokens = _as_float(getattr(cfg, "max_tokens", None), None)
        if output_tokens is None:
            output_tokens = _as_float(os.environ.get("MFE_OUTPUT_MAX_TOKENS"), 256.0) or 256.0
        return max(self.min_runtime, self.default_op_time + self.output_token_time * output_tokens)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _query_create_time(self, q: Optional[Query]) -> float:
        if q is None:
            return 0.0
        return float(getattr(q, "create_time", 0.0) or 0.0)

    def _query_complete(self, q: Query, get_dag: OpsGetter) -> bool:
        try:
            _, _, end_ops = get_dag(q.template)
        except Exception:
            return False
        done = set(getattr(q, "op_output", {}) or {})
        end_ids = {str(op.id) for op in end_ops}
        return bool(end_ids) and end_ids <= done

    def _is_start_op(self, q: Query, op: Operator, get_dag: OpsGetter) -> bool:
        try:
            metrics = self._template_metrics(q.template, get_dag)
        except Exception:
            return not getattr(op, "input_ops", None)
        return str(op.id) in metrics.get("start_ids", set())