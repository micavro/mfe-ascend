"""RH-SAIL online scheduler for multi-request MFE execution.

RH-SAIL combines two complementary scheduling layers:

- SAIL contributes per-workflow criticality, preferred placement, and optional
  state-reuse affinity.
- RHRS contributes diverse candidate construction and limited-horizon rollout
  over the currently visible ready operators.

The scheduler adds runtime controls that neither admission-time SAIL plans nor
plain request aging provide: bounded active-workflow admission, progress
commitment after a workflow starts, inter-op gap protection, service-stretch
protection, and an online operator runtime model.

It intentionally orders individual ``(query, operator)`` tasks.  It does not
introduce an outer query batch or change the vLLM worker interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger
import math
import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from mfe.components import Operator, Query
from mfe.optimizers.sailp import SAILPScheduler, SchedulePlan


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


@dataclass
class _RuntimeStat:
    duration: float
    input_tokens: float
    output_tokens: float
    observations: int = 1


@dataclass(frozen=True)
class RHSailTaskScore:
    uid: str
    op_id: str
    score: float
    runtime: float
    critical_path: float
    unlock_work: float
    gap_age: float
    service_stretch: float
    affinity: float
    admission: float
    commitment: float
    is_start_op: bool
    is_end_op: bool


class RHSailReadyScheduler:
    """Online completion-oriented scheduler with SAIL structural guidance."""

    def __init__(
        self,
        num_workers: int,
        *,
        use_sail_guidance: Optional[bool] = None,
    ) -> None:
        self.num_workers = max(1, int(num_workers))

        raw_limit = _env_int("MFE_RHSAIL_ACTIVE_DAG_LIMIT", 0)
        self.active_dag_limit = raw_limit if raw_limit > 0 else int(
            math.ceil(_env_float("MFE_RHSAIL_ACTIVE_DAG_FACTOR", 3.0) * self.num_workers)
        )
        self.active_dag_limit = max(1, self.active_dag_limit)
        self.admission_max_wait = max(1.0, _env_float("MFE_RHSAIL_ADMISSION_MAX_WAIT", 300.0))

        self.candidate_k = max(1, _env_int("MFE_RHSAIL_CANDIDATE_K", 12))
        self.rollout_horizon = max(1, _env_int("MFE_RHSAIL_ROLLOUT_HORIZON", 3))
        self.commitment_ops = max(1, _env_int("MFE_RHSAIL_COMMITMENT_OPS", 2))

        self.soft_gap_seconds = max(1.0, _env_float("MFE_RHSAIL_SOFT_GAP_SECONDS", 30.0))
        self.hard_gap_seconds = max(
            self.soft_gap_seconds,
            _env_float("MFE_RHSAIL_HARD_GAP_SECONDS", 180.0),
        )
        self.commitment_grace_seconds = max(
            1.0,
            _env_float("MFE_RHSAIL_COMMITMENT_GRACE_SECONDS", self.soft_gap_seconds),
        )
        self.soft_stretch = max(1.0, _env_float("MFE_RHSAIL_SOFT_STRETCH", 2.0))
        self.hard_stretch = max(
            self.soft_stretch,
            _env_float("MFE_RHSAIL_HARD_STRETCH", 6.0),
        )

        # Candidate pruning weights.  All features are normalized per workflow,
        # so these defaults do not depend on a particular benchmark family.
        self.w_cp = _env_float("MFE_RHSAIL_W_CP", 0.18)
        self.w_unlock = _env_float("MFE_RHSAIL_W_UNLOCK", 0.12)
        self.w_finish = _env_float("MFE_RHSAIL_W_FINISH", 0.12)
        self.w_gap = _env_float("MFE_RHSAIL_W_GAP", 0.18)
        self.w_stretch = _env_float("MFE_RHSAIL_W_STRETCH", 0.18)
        self.w_commitment = _env_float("MFE_RHSAIL_W_COMMITMENT", 0.10)
        self.w_affinity = _env_float("MFE_RHSAIL_W_AFFINITY", 0.07)
        self.w_admission = _env_float("MFE_RHSAIL_W_ADMISSION", 0.08)
        self.w_short = _env_float("MFE_RHSAIL_W_SHORT", 0.05)

        # Rollout objective.  The unfinished-request area is the ACT term;
        # service and gap pressure make the policy completion-fair.
        self.eta_flow = _env_float("MFE_RHSAIL_ETA_FLOW", 1.0)
        self.eta_service = _env_float("MFE_RHSAIL_ETA_SERVICE", 0.30)
        self.eta_gap = _env_float("MFE_RHSAIL_ETA_GAP", 0.30)
        self.eta_ready = _env_float("MFE_RHSAIL_ETA_READY", 0.05)
        self.eta_terminal_cp = _env_float("MFE_RHSAIL_ETA_TERMINAL_CP", 0.20)
        self.eta_affinity = _env_float("MFE_RHSAIL_ETA_AFFINITY", 0.05)

        # Runtime model.  Completed real operators update an EWMA keyed by
        # template, operator, and prompt-length bucket.
        self.default_op_time = _env_float("MFE_RHSAIL_DEFAULT_OP_TIME", 0.2)
        self.input_token_time = _env_float("MFE_RHSAIL_INPUT_TOKEN_TIME", 0.00015)
        self.output_token_time = _env_float("MFE_RHSAIL_OUTPUT_TOKEN_TIME", 0.004)
        self.output_fraction = min(
            1.0,
            max(0.0, _env_float("MFE_RHSAIL_OUTPUT_FRACTION", 0.25)),
        )
        self.runtime_ewma_alpha = min(
            1.0,
            max(0.01, _env_float("MFE_RHSAIL_RUNTIME_EWMA_ALPHA", 0.25)),
        )
        self.min_runtime = max(_EPS, _env_float("MFE_RHSAIL_MIN_RUNTIME", 0.05))

        if use_sail_guidance is None:
            use_sail_guidance = _env_bool("MFE_RHSAIL_USE_SAIL_GUIDANCE", True)
        self.use_sail_guidance = bool(use_sail_guidance)
        self._sail_planner = (
            SAILPScheduler(num_executors=self.num_workers, executor_ids=list(range(self.num_workers)))
            if self.use_sail_guidance
            else None
        )

        self.verbose = _env_bool("MFE_RHSAIL_VERBOSE", False)
        self._structure_cache: Dict[str, Dict[str, Any]] = {}
        self._sail_plan_cache: Dict[str, Optional[SchedulePlan]] = {}
        self._runtime_stats: Dict[Tuple[str, str, int], _RuntimeStat] = {}
        self._dynamic_cp_cache: Dict[Tuple[str, Tuple[str, ...]], Dict[str, float]] = {}

        self._decisions = 0
        self._admission_throttles = 0
        self._emergency_decisions = 0
        self._max_gap_seen = 0.0
        self._max_stretch_seen = 0.0

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
        if not ready:
            return []

        now = time.perf_counter() if now is None else float(now)
        self._decisions += 1

        filtered = self._apply_admission_control(
            ready,
            requests=requests,
            inflight_tasks=inflight_tasks,
            get_dag=get_dag,
            now=now,
        )
        if not filtered:
            filtered = ready

        scored = [
            self._score_task(
                task,
                worker_id=worker_id,
                requests=requests,
                inflight_tasks=inflight_tasks,
                get_dag=get_dag,
                now=now,
            )
            for task in filtered
            if task[0] in requests
        ]
        if not scored:
            return filtered

        emergency = [s for s in scored if self._emergency_severity(s) >= 1.0]
        if emergency:
            self._emergency_decisions += 1
            scored = emergency

        candidates = self._build_diverse_candidates(scored)
        task_by_key = {(uid, str(op.id)): (uid, op) for uid, op in filtered}
        final: List[Tuple[float, float, float, ReadyTask, RHSailTaskScore]] = []
        for score_obj in candidates:
            task = task_by_key.get((score_obj.uid, score_obj.op_id))
            if task is None:
                continue
            rollout_cost = self._rollout_cost(
                first_task=task,
                worker_id=worker_id,
                ready=filtered,
                requests=requests,
                inflight_tasks=inflight_tasks,
                get_dag=get_dag,
                now=now,
            )
            final.append(
                (
                    rollout_cost,
                    -self._emergency_severity(score_obj),
                    -score_obj.score,
                    task,
                    score_obj,
                )
            )

        if not final:
            return filtered

        final.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                self._query_create_time(requests.get(item[3][0])),
                item[3][0],
                item[3][1].id,
            )
        )
        candidate_tasks = [item[3] for item in final]
        candidate_keys = {(uid, str(op.id)) for uid, op in candidate_tasks}
        rest = sorted(
            (
                (score.score, task_by_key[(score.uid, score.op_id)])
                for score in scored
                if (score.uid, score.op_id) not in candidate_keys
                and (score.uid, score.op_id) in task_by_key
            ),
            key=lambda item: (
                -item[0],
                self._query_create_time(requests.get(item[1][0])),
                item[1][0],
                item[1][1].id,
            ),
        )
        ordered = candidate_tasks + [task for _, task in rest]

        if self.verbose and ordered:
            best = final[0][4]
            print(
                "[RH-SAIL] worker={} choose uid={} op={} score={:.4f} rollout={:.4f} "
                "gap={:.1f}s stretch={:.2f} affinity={:.2f} commitment={:.2f}".format(
                    worker_id,
                    best.uid[:8],
                    best.op_id,
                    best.score,
                    final[0][0],
                    best.gap_age,
                    best.service_stretch,
                    best.affinity,
                    best.commitment,
                ),
                flush=True,
            )

        return ordered

    def observe_completion(
        self,
        q: Query,
        op_id: str,
        *,
        worker_id: int,
        benchmark: Sequence[Any],
        metrics: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if len(benchmark) < 2:
            return
        start = _as_float(benchmark[0], None)
        end = _as_float(benchmark[1], None)
        if start is None or end is None or end <= start:
            return

        metrics = metrics or {}
        input_tokens = _as_float(metrics.get("input_tokens"), None)
        if input_tokens is None:
            input_tokens = float(getattr(q, "input_len_est_tokens", 0) or 0)
        output_tokens = _as_float(metrics.get("output_tokens"), 0.0) or 0.0
        bucket = self._token_bucket(input_tokens)
        key = (str(q.template), str(op_id), bucket)
        duration = max(self.min_runtime, end - start)
        current = self._runtime_stats.get(key)
        if current is None:
            self._runtime_stats[key] = _RuntimeStat(duration, input_tokens, output_tokens)
        else:
            alpha = self.runtime_ewma_alpha
            current.duration = alpha * duration + (1.0 - alpha) * current.duration
            current.input_tokens = alpha * input_tokens + (1.0 - alpha) * current.input_tokens
            current.output_tokens = alpha * output_tokens + (1.0 - alpha) * current.output_tokens
            current.observations += 1
        # The completing query moves to a new DAG state.  Other active-query
        # path estimates remain stable until their own progress changes; direct
        # task runtime scoring still uses the latest EWMA immediately.
        for cache_key in [item for item in self._dynamic_cp_cache if item[0] == q.id]:
            self._dynamic_cp_cache.pop(cache_key, None)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": "rhsail",
            "active_dag_limit": self.active_dag_limit,
            "candidate_k": self.candidate_k,
            "rollout_horizon": self.rollout_horizon,
            "commitment_ops": self.commitment_ops,
            "soft_gap_seconds": self.soft_gap_seconds,
            "hard_gap_seconds": self.hard_gap_seconds,
            "soft_stretch": self.soft_stretch,
            "hard_stretch": self.hard_stretch,
            "use_sail_guidance": self.use_sail_guidance,
            "runtime_model_buckets": len(self._runtime_stats),
            "decisions": self._decisions,
            "admission_throttles": self._admission_throttles,
            "emergency_decisions": self._emergency_decisions,
            "max_gap_seen": self._max_gap_seen,
            "max_stretch_seen": self._max_stretch_seen,
        }

    # ------------------------------------------------------------------
    # Admission and hard progress guards
    # ------------------------------------------------------------------

    def _apply_admission_control(
        self,
        ready: List[ReadyTask],
        *,
        requests: Mapping[str, Query],
        inflight_tasks: Set[Tuple[str, str]],
        get_dag: OpsGetter,
        now: float,
    ) -> List[ReadyTask]:
        active = self._active_query_ids(requests, inflight_tasks, get_dag)
        active_ready = [task for task in ready if task[0] in active]
        active_work = len(active_ready) + sum(1 for uid, _ in inflight_tasks if uid in active)

        start_tasks = [
            task
            for task in ready
            if task[0] not in active
            and self._is_start_op(requests.get(task[0]), task[1], get_dag)
        ]
        ranked_starts = sorted(
            start_tasks,
            key=lambda task: (
                -self._admission_value(
                    requests[task[0]],
                    now,
                    float(self._structure(requests[task[0]].template, get_dag)["template_cp"]),
                ),
                self._query_create_time(requests.get(task[0])),
                task[0],
                task[1].id,
            ),
        )

        # Admit only enough roots to fill the active pool.  Returning every
        # waiting root recreates the breadth-first queue explosion that RH-SAIL
        # is intended to prevent.
        if len(active) < self.active_dag_limit:
            slots = self.active_dag_limit - len(active)
            return active_ready + ranked_starts[:slots]

        # The limit remains work-conserving when active workflows temporarily
        # expose too little parallel work.
        if active_work < self.num_workers:
            needed = self.num_workers - active_work
            return active_ready + ranked_starts[:needed]

        starved = [
            task
            for task in ranked_starts
            if now - self._query_create_time(requests.get(task[0])) >= self.admission_max_wait
        ]

        self._admission_throttles += 1
        if active_ready:
            if starved:
                oldest = min(
                    starved,
                    key=lambda item: (
                        self._query_create_time(requests.get(item[0])),
                        item[0],
                        item[1].id,
                    ),
                )
                return active_ready + [oldest]
            return active_ready

        # Defensive fallback for blocked or dynamically changing workflows.
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
        inflight_uids = {uid for uid, _ in inflight_tasks}
        active: Set[str] = set()
        for uid, q in requests.items():
            if getattr(q, "status", None) == "error" or self._query_complete(q, get_dag):
                continue
            if uid in inflight_uids or bool(getattr(q, "op_output", None)):
                active.add(uid)
        return active

    def _emergency_severity(self, score: RHSailTaskScore) -> float:
        if score.is_start_op:
            return 0.0
        gap = score.gap_age / max(self.hard_gap_seconds, _EPS)
        stretch = score.service_stretch / max(self.hard_stretch, _EPS)
        commitment = 0.0
        if score.commitment > 0.0:
            commitment = score.gap_age / max(self.commitment_grace_seconds, _EPS)
        return max(gap, stretch, commitment)

    # ------------------------------------------------------------------
    # Candidate construction and scoring
    # ------------------------------------------------------------------

    def _score_task(
        self,
        task: ReadyTask,
        *,
        worker_id: int,
        requests: Mapping[str, Query],
        inflight_tasks: Set[Tuple[str, str]],
        get_dag: OpsGetter,
        now: float,
    ) -> RHSailTaskScore:
        uid, op = task
        q = requests[uid]
        structure = self._structure(q.template, get_dag)
        done = set(getattr(q, "op_output", {}) or {})
        op_id = str(op.id)

        runtime = self._estimate_runtime(q, op)
        critical_path = self._remaining_cp_from_op(q, op, done, get_dag)
        template_cp = max(_EPS, float(structure["template_cp"]))
        cp_norm = min(1.0, critical_path / template_cp)

        unlock_work = self._unlock_work(q, op, done, get_dag)
        unlock_norm = min(1.0, unlock_work / template_cp)
        is_start = op_id in structure["start_ids"] and not done
        is_end = op_id in structure["end_ids"]

        has_inflight = any(xuid == uid for xuid, _ in inflight_tasks)
        gap_age = self._gap_age(q, now, has_inflight=has_inflight)
        remaining_cp = self._query_remaining_cp(q, done, get_dag)
        service_stretch = self._service_stretch(q, now, remaining_cp, template_cp)
        self._max_gap_seen = max(self._max_gap_seen, gap_age)
        self._max_stretch_seen = max(self._max_stretch_seen, service_stretch)

        gap_norm = min(1.0, gap_age / max(self.soft_gap_seconds, _EPS))
        stretch_norm = min(
            1.0,
            max(0.0, service_stretch - 1.0) / max(self.soft_stretch - 1.0, _EPS),
        )
        commitment = self._commitment_value(q, has_inflight=has_inflight)
        affinity = self._state_affinity(q, op, worker_id, get_dag)
        admission = self._admission_value(q, now, template_cp) if is_start else 0.0
        short = 1.0 - min(1.0, runtime / template_cp)

        score = (
            self.w_cp * cp_norm
            + self.w_unlock * unlock_norm
            + self.w_finish * float(is_end)
            + self.w_gap * gap_norm
            + self.w_stretch * stretch_norm
            + self.w_commitment * commitment
            + self.w_affinity * affinity
            + self.w_admission * admission
            + self.w_short * short
        )

        return RHSailTaskScore(
            uid=uid,
            op_id=op_id,
            score=score,
            runtime=runtime,
            critical_path=critical_path,
            unlock_work=unlock_work,
            gap_age=gap_age,
            service_stretch=service_stretch,
            affinity=affinity,
            admission=admission,
            commitment=commitment,
            is_start_op=is_start,
            is_end_op=is_end,
        )

    def _build_diverse_candidates(self, scored: List[RHSailTaskScore]) -> List[RHSailTaskScore]:
        if len(scored) <= self.candidate_k:
            return sorted(scored, key=lambda item: (-item.score, item.uid, item.op_id))

        chosen: List[RHSailTaskScore] = []
        chosen_keys: Set[Tuple[str, str]] = set()

        def add_best(key: Callable[[RHSailTaskScore], Tuple[Any, ...]]) -> None:
            if len(chosen) >= self.candidate_k:
                return
            item = max(scored, key=key)
            item_key = (item.uid, item.op_id)
            if item_key not in chosen_keys:
                chosen.append(item)
                chosen_keys.add(item_key)

        # RHRS-style candidate diversity: fairness, progress, shortness,
        # completion, state affinity, and admission each get representation.
        add_best(lambda s: (s.gap_age, s.score))
        add_best(lambda s: (s.service_stretch, s.score))
        add_best(lambda s: (s.commitment, s.gap_age, s.score))
        add_best(lambda s: (float(s.is_end_op), -s.runtime, s.score))
        add_best(lambda s: (s.critical_path + s.unlock_work, s.score))
        add_best(lambda s: (s.affinity, s.score))
        add_best(lambda s: (s.admission, -s.runtime, s.score))
        add_best(lambda s: (-s.runtime, s.score))

        for item in sorted(scored, key=lambda s: (-s.score, s.uid, s.op_id)):
            if len(chosen) >= self.candidate_k:
                break
            item_key = (item.uid, item.op_id)
            if item_key not in chosen_keys:
                chosen.append(item)
                chosen_keys.add(item_key)
        return chosen

    # ------------------------------------------------------------------
    # Limited-horizon rollout
    # ------------------------------------------------------------------

    def _rollout_cost(
        self,
        *,
        first_task: ReadyTask,
        worker_id: int,
        ready: List[ReadyTask],
        requests: Mapping[str, Query],
        inflight_tasks: Set[Tuple[str, str]],
        get_dag: OpsGetter,
        now: float,
    ) -> float:
        shadow_done = {
            uid: set(getattr(q, "op_output", {}) or {})
            for uid, q in requests.items()
        }
        shadow_ready = list(ready)
        shadow_first_start = {
            uid: self._first_start(q)
            for uid, q in requests.items()
        }
        shadow_last_progress = {
            uid: self._last_completion(q)
            for uid, q in requests.items()
        }
        shadow_active_uids = {
            uid for uid, first_start in shadow_first_start.items()
            if first_start is not None
        }
        real_inflight_uids = {uid for uid, _ in inflight_tasks}
        unfinished = self._shadow_unfinished_count(requests, shadow_done, get_dag)

        total = 0.0
        t = now
        chosen: Optional[ReadyTask] = first_task
        first_affinity = self._state_affinity(
            requests[first_task[0]], first_task[1], worker_id, get_dag
        )

        for step in range(self.rollout_horizon):
            if chosen is None:
                break
            uid, op = chosen
            q = requests.get(uid)
            if q is None:
                break

            runtime = self._estimate_runtime(q, op)
            op_id = str(op.id)
            shadow_ready = [
                task for task in shadow_ready
                if not (task[0] == uid and str(task[1].id) == op_id)
            ]
            if shadow_first_start.get(uid) is None:
                shadow_first_start[uid] = t
                shadow_active_uids.add(uid)
            structure = self._structure(q.template, get_dag)
            was_complete = structure["end_ids"] <= shadow_done.get(uid, set())
            t += runtime
            shadow_done.setdefault(uid, set()).add(op_id)
            shadow_last_progress[uid] = t
            is_complete = structure["end_ids"] <= shadow_done[uid]
            if is_complete and not was_complete:
                unfinished = max(0, unfinished - 1)
            self._unlock_shadow_successors(
                uid,
                op,
                shadow_ready=shadow_ready,
                shadow_done=shadow_done,
                requests=requests,
                get_dag=get_dag,
            )

            service_pressure, gap_pressure = self._shadow_progress_pressure(
                requests,
                shadow_done=shadow_done,
                shadow_first_start=shadow_first_start,
                shadow_last_progress=shadow_last_progress,
                active_uids=shadow_active_uids,
                inflight_uids=real_inflight_uids,
                get_dag=get_dag,
                now=t,
            )
            total += runtime * (
                self.eta_flow * unfinished
                * (1.0 + self.eta_service * service_pressure + self.eta_gap * gap_pressure)
                + self.eta_ready * len(shadow_ready)
            )

            if step + 1 >= self.rollout_horizon or not shadow_ready:
                break
            chosen = self._shadow_best_task(
                shadow_ready,
                worker_id=worker_id,
                requests=requests,
                shadow_done=shadow_done,
                shadow_first_start=shadow_first_start,
                shadow_last_progress=shadow_last_progress,
                get_dag=get_dag,
                now=t,
            )

        total += self.eta_terminal_cp * self._shadow_remaining_cp(
            requests, shadow_done, shadow_active_uids, get_dag
        )
        total -= self.eta_affinity * first_affinity
        return total

    def _shadow_best_task(
        self,
        shadow_ready: List[ReadyTask],
        *,
        worker_id: int,
        requests: Mapping[str, Query],
        shadow_done: Mapping[str, Set[str]],
        shadow_first_start: Mapping[str, Optional[float]],
        shadow_last_progress: Mapping[str, Optional[float]],
        get_dag: OpsGetter,
        now: float,
    ) -> Optional[ReadyTask]:
        best: Optional[ReadyTask] = None
        best_score = -math.inf
        for uid, op in shadow_ready:
            q = requests.get(uid)
            if q is None:
                continue
            structure = self._structure(q.template, get_dag)
            done = shadow_done.get(uid, set())
            template_cp = max(_EPS, float(structure["template_cp"]))
            runtime = self._estimate_runtime(q, op)
            cp = self._remaining_cp_from_op(q, op, set(done), get_dag)
            unlock = self._unlock_work(q, op, set(done), get_dag)
            remaining = self._query_remaining_cp(q, set(done), get_dag)
            first_start = shadow_first_start.get(uid)
            if first_start is None:
                stretch = 0.0
                admission = self._admission_value(q, now, template_cp)
            else:
                stretch = (max(0.0, now - first_start) + remaining) / template_cp
                admission = 0.0
            last = shadow_last_progress.get(uid)
            gap = max(0.0, now - last) if last is not None else 0.0
            affinity = self._state_affinity(q, op, worker_id, get_dag)
            score = (
                self.w_cp * min(1.0, cp / template_cp)
                + self.w_unlock * min(1.0, unlock / template_cp)
                + self.w_finish * float(str(op.id) in structure["end_ids"])
                + self.w_gap * min(1.0, gap / self.soft_gap_seconds)
                + self.w_stretch * min(1.0, max(0.0, stretch - 1.0))
                + self.w_affinity * affinity
                + self.w_admission * admission
                + self.w_short * (1.0 - min(1.0, runtime / template_cp))
            )
            if score > best_score:
                best_score = score
                best = (uid, op)
        return best

    def _unlock_shadow_successors(
        self,
        uid: str,
        op: Operator,
        *,
        shadow_ready: List[ReadyTask],
        shadow_done: Mapping[str, Set[str]],
        requests: Mapping[str, Query],
        get_dag: OpsGetter,
    ) -> None:
        q = requests.get(uid)
        if q is None:
            return
        try:
            ops, _, _ = get_dag(q.template)
        except Exception:
            return
        done = shadow_done.get(uid, set())
        for child in getattr(op, "output_ops", []) or []:
            child_id = str(child.id)
            if child_id not in ops or child_id in done:
                continue
            child_op = ops[child_id]
            if not all(str(parent.id) in done for parent in getattr(child_op, "input_ops", []) or []):
                continue
            if not any(xuid == uid and str(xop.id) == child_id for xuid, xop in shadow_ready):
                shadow_ready.append((uid, child_op))

    def _shadow_unfinished_count(
        self,
        requests: Mapping[str, Query],
        shadow_done: Mapping[str, Set[str]],
        get_dag: OpsGetter,
    ) -> int:
        total = 0
        for uid, q in requests.items():
            if getattr(q, "status", None) == "error":
                continue
            structure = self._structure(q.template, get_dag)
            if not structure["end_ids"] <= shadow_done.get(uid, set()):
                total += 1
        return total

    def _shadow_progress_pressure(
        self,
        requests: Mapping[str, Query],
        *,
        shadow_done: Mapping[str, Set[str]],
        shadow_first_start: Mapping[str, Optional[float]],
        shadow_last_progress: Mapping[str, Optional[float]],
        active_uids: Set[str],
        inflight_uids: Set[str],
        get_dag: OpsGetter,
        now: float,
    ) -> Tuple[float, float]:
        service_values: List[float] = []
        gap_values: List[float] = []
        for uid in active_uids:
            q = requests.get(uid)
            if q is None:
                continue
            first_start = shadow_first_start.get(uid)
            if first_start is None:
                continue
            structure = self._structure(q.template, get_dag)
            if structure["end_ids"] <= shadow_done.get(uid, set()):
                continue
            template_cp = max(_EPS, float(structure["template_cp"]))
            remaining = self._shadow_query_remaining_cp(q, shadow_done.get(uid, set()), get_dag)
            stretch = (max(0.0, now - first_start) + remaining) / template_cp
            stretch_excess = max(0.0, stretch - self.soft_stretch) / max(
                self.hard_stretch - self.soft_stretch,
                _EPS,
            )
            service_values.append(min(4.0, stretch_excess) ** 2)

            last = shadow_last_progress.get(uid)
            if last is not None and uid not in inflight_uids:
                gap_excess = max(0.0, now - last - self.soft_gap_seconds) / max(
                    self.hard_gap_seconds - self.soft_gap_seconds,
                    _EPS,
                )
                gap_values.append(min(4.0, gap_excess) ** 2)

        service = sum(service_values) / max(1, len(service_values))
        gap = sum(gap_values) / max(1, len(gap_values))
        return service, gap

    def _shadow_remaining_cp(
        self,
        requests: Mapping[str, Query],
        shadow_done: Mapping[str, Set[str]],
        active_uids: Set[str],
        get_dag: OpsGetter,
    ) -> float:
        return sum(
            self._shadow_query_remaining_cp(q, shadow_done.get(uid, set()), get_dag)
            for uid in active_uids
            if (q := requests.get(uid)) is not None
            and getattr(q, "status", None) != "error"
        )

    # ------------------------------------------------------------------
    # DAG, progress, and affinity features
    # ------------------------------------------------------------------

    def _structure(self, template: str, get_dag: OpsGetter) -> Dict[str, Any]:
        key = str(template)
        cached = self._structure_cache.get(key)
        if cached is not None:
            return cached

        ops, start_ops, end_ops = get_dag(template)
        static_cp: Dict[str, float] = {}
        visiting: Set[str] = set()

        def dfs(op: Operator) -> float:
            op_id = str(op.id)
            if op_id in static_cp:
                return static_cp[op_id]
            if op_id in visiting:
                return self.default_op_time
            visiting.add(op_id)
            children = [child for child in getattr(op, "output_ops", []) or [] if str(child.id) in ops]
            own = self._estimate_static_runtime(op)
            static_cp[op_id] = own + max((dfs(child) for child in children), default=0.0)
            visiting.remove(op_id)
            return static_cp[op_id]

        for op in ops.values():
            dfs(op)

        start_ids = {str(op.id) for op in start_ops}
        end_ids = {str(op.id) for op in end_ops}
        template_cp = max(
            (static_cp.get(op_id, 0.0) for op_id in start_ids),
            default=max(static_cp.values(), default=1.0),
        )
        result = {
            "ops": ops,
            "start_ids": start_ids,
            "end_ids": end_ids,
            "static_cp": static_cp,
            "template_cp": max(_EPS, template_cp),
            "total_ops": len(ops),
        }
        self._structure_cache[key] = result
        return result

    def _remaining_cp_from_op(
        self,
        q: Query,
        op: Operator,
        done: Set[str],
        get_dag: OpsGetter,
    ) -> float:
        return self._dynamic_cp_map(q, done, get_dag).get(
            str(op.id),
            self._estimate_runtime(q, op),
        )

    def _query_remaining_cp(
        self,
        q: Query,
        done: Set[str],
        get_dag: OpsGetter,
    ) -> float:
        cp = self._dynamic_cp_map(q, done, get_dag)
        return max((value for op_id, value in cp.items() if op_id not in done), default=0.0)

    def _dynamic_cp_map(
        self,
        q: Query,
        done: Set[str],
        get_dag: OpsGetter,
    ) -> Dict[str, float]:
        cache_key = (str(q.id), tuple(sorted(done)))
        cached = self._dynamic_cp_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            ops, _, _ = get_dag(q.template)
        except Exception:
            return {}

        memo: Dict[str, float] = {}

        def dfs(node: Operator) -> float:
            node_id = str(node.id)
            if node_id in done:
                return 0.0
            if node_id in memo:
                return memo[node_id]
            children = [child for child in getattr(node, "output_ops", []) or [] if str(child.id) in ops]
            memo[node_id] = self._estimate_runtime(q, node) + max(
                (dfs(child) for child in children),
                default=0.0,
            )
            return memo[node_id]

        for node in ops.values():
            dfs(node)
        self._dynamic_cp_cache[cache_key] = memo
        return memo

    def _shadow_query_remaining_cp(
        self,
        q: Query,
        done: Set[str],
        get_dag: OpsGetter,
    ) -> float:
        structure = self._structure(q.template, get_dag)
        remaining = [
            cp for op_id, cp in structure["static_cp"].items()
            if op_id not in done
        ]
        return max(remaining, default=0.0)

    def _unlock_work(
        self,
        q: Query,
        op: Operator,
        done: Set[str],
        get_dag: OpsGetter,
    ) -> float:
        structure = self._structure(q.template, get_dag)
        value = 0.0
        for child in getattr(op, "output_ops", []) or []:
            child_id = str(child.id)
            other_parents_done = all(
                str(parent.id) == str(op.id) or str(parent.id) in done
                for parent in getattr(child, "input_ops", []) or []
            )
            if other_parents_done:
                value += float(structure["static_cp"].get(child_id, 0.0))
        return value

    def _state_affinity(
        self,
        q: Query,
        op: Operator,
        worker_id: int,
        get_dag: OpsGetter,
    ) -> float:
        values: List[Tuple[float, float]] = []
        plan = self._sail_plan(q.template, get_dag)
        if plan is not None:
            step = plan.steps.get(str(op.id))
            if step is not None:
                values.append((0.45, float(step.worker_id == worker_id)))
                if step.reuse_from:
                    source_worker = q.worker_assignments.get(step.reuse_from)
                    if source_worker is not None:
                        values.append((0.20, float(int(source_worker) == worker_id)))

        parents = list(getattr(op, "input_ops", []) or [])
        assigned_parents = [
            q.worker_assignments.get(str(parent.id))
            for parent in parents
            if str(parent.id) in q.worker_assignments
        ]
        if assigned_parents:
            local_fraction = sum(int(worker) == worker_id for worker in assigned_parents) / len(assigned_parents)
            values.append((0.35, local_fraction))

        if not values:
            return 0.0
        total_weight = sum(weight for weight, _ in values)
        return sum(weight * value for weight, value in values) / max(total_weight, _EPS)

    def _sail_plan(self, template: str, get_dag: OpsGetter) -> Optional[SchedulePlan]:
        if self._sail_planner is None:
            return None
        key = str(template)
        if key in self._sail_plan_cache:
            return self._sail_plan_cache[key]
        try:
            ops, _, _ = get_dag(template)
            plan = self._sail_planner.plan(ops)
        except Exception as exc:
            logger.warning("RH-SAIL disabled SAIL guidance for %s: %s", template, exc)
            plan = None
        self._sail_plan_cache[key] = plan
        return plan

    # ------------------------------------------------------------------
    # Runtime estimation
    # ------------------------------------------------------------------

    def _estimate_runtime(self, q: Query, op: Operator) -> float:
        spec = _safe_mapping(getattr(op, "sailp", None))
        for key in ("cold_time", "estimated_time", "duration", "cost"):
            value = _as_float(spec.get(key), None)
            if value is not None:
                return max(self.min_runtime, value)

        prompt_tokens = self._estimated_prompt_tokens(q, op)
        bucket = self._token_bucket(prompt_tokens)
        stat = self._runtime_stats.get((str(q.template), str(op.id), bucket))
        if stat is not None:
            return max(self.min_runtime, stat.duration)

        cfg = getattr(op, "model_config", None)
        max_output = _as_float(getattr(cfg, "max_tokens", None), None)
        if max_output is None:
            max_output = _as_float(os.environ.get("MFE_OUTPUT_MAX_TOKENS"), 256.0) or 256.0
        expected_output = max(1.0, max_output * self.output_fraction)
        return max(
            self.min_runtime,
            self.default_op_time
            + self.input_token_time * prompt_tokens
            + self.output_token_time * expected_output,
        )

    def _estimate_static_runtime(self, op: Operator) -> float:
        spec = _safe_mapping(getattr(op, "sailp", None))
        for key in ("cold_time", "estimated_time", "duration", "cost"):
            value = _as_float(spec.get(key), None)
            if value is not None:
                return max(self.min_runtime, value)
        cfg = getattr(op, "model_config", None)
        max_output = _as_float(getattr(cfg, "max_tokens", None), None)
        if max_output is None:
            max_output = _as_float(os.environ.get("MFE_OUTPUT_MAX_TOKENS"), 256.0) or 256.0
        return max(
            self.min_runtime,
            self.default_op_time + self.output_token_time * max(1.0, max_output * self.output_fraction),
        )

    def _estimated_prompt_tokens(self, q: Query, op: Operator) -> float:
        tokens = float(getattr(q, "input_len_est_tokens", 0) or 0)
        ancestor_ids: Set[str] = set()

        def visit(node: Operator) -> None:
            for parent in getattr(node, "input_ops", []) or []:
                parent_id = str(parent.id)
                if parent_id in ancestor_ids:
                    continue
                ancestor_ids.add(parent_id)
                visit(parent)

        visit(op)
        for ancestor_id in ancestor_ids:
            output = (getattr(q, "op_output", {}) or {}).get(ancestor_id)
            if output:
                tokens += max(1.0, len(str(output)) / 4.0)
        return max(1.0, tokens)

    @staticmethod
    def _token_bucket(tokens: float) -> int:
        return int(math.floor(math.log2(max(1.0, float(tokens)))))

    # ------------------------------------------------------------------
    # Query progress helpers
    # ------------------------------------------------------------------

    def _query_create_time(self, q: Optional[Query]) -> float:
        if q is None:
            return 0.0
        return float(getattr(q, "create_time", 0.0) or 0.0)

    def _first_start(self, q: Query) -> Optional[float]:
        benchmark = getattr(q, "benchmark", {}) or {}
        starts = [
            _as_float(value[0], None)
            for value in benchmark.values()
            if isinstance(value, (list, tuple)) and len(value) >= 2
        ]
        valid = [value for value in starts if value is not None]
        return min(valid) if valid else None

    def _last_completion(self, q: Query) -> Optional[float]:
        benchmark = getattr(q, "benchmark", {}) or {}
        ends = [
            _as_float(value[1], None)
            for value in benchmark.values()
            if isinstance(value, (list, tuple)) and len(value) >= 2
        ]
        valid = [value for value in ends if value is not None]
        return max(valid) if valid else None

    def _gap_age(self, q: Query, now: float, *, has_inflight: bool) -> float:
        if has_inflight:
            return 0.0
        last = self._last_completion(q)
        return max(0.0, now - last) if last is not None else 0.0

    def _service_stretch(
        self,
        q: Query,
        now: float,
        remaining_cp: float,
        template_cp: float,
    ) -> float:
        first = self._first_start(q)
        if first is None:
            return 0.0
        return (max(0.0, now - first) + remaining_cp) / max(template_cp, _EPS)

    def _commitment_value(self, q: Query, *, has_inflight: bool) -> float:
        if has_inflight or not getattr(q, "op_output", None):
            return 0.0
        completed = int(getattr(q, "step", 0) or 0)
        remaining = max(0, self.commitment_ops - completed)
        return remaining / self.commitment_ops

    def _admission_value(self, q: Query, now: float, template_cp: float) -> float:
        wait = max(0.0, now - self._query_create_time(q))
        # Monotonic transform of HRRN = 1 + wait / service.
        return wait / max(wait + template_cp, _EPS)

    def _query_complete(self, q: Query, get_dag: OpsGetter) -> bool:
        try:
            structure = self._structure(q.template, get_dag)
        except Exception:
            return False
        return bool(structure["end_ids"]) and structure["end_ids"] <= set(
            getattr(q, "op_output", {}) or {}
        )

    def _is_start_op(
        self,
        q: Optional[Query],
        op: Operator,
        get_dag: OpsGetter,
    ) -> bool:
        if q is None:
            return not getattr(op, "input_ops", None)
        try:
            structure = self._structure(q.template, get_dag)
        except Exception:
            return not getattr(op, "input_ops", None)
        return str(op.id) in structure["start_ids"] and not getattr(q, "op_output", None)
