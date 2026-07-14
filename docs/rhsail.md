# RH-SAIL Scheduler

RH-SAIL is an online scheduler for concurrent MFE workflows. It combines:

- SAIL workflow guidance: critical-path structure, preferred worker placement,
  and optional reuse affinity.
- RHRS runtime guidance: diverse ready-op candidates, pruning, and
  limited-horizon rollout over cumulative completion cost.
- Completion controls: active-workflow admission, progress commitment,
  inter-op gap protection, service-stretch protection, and online runtime
  calibration from completed operators.

RH-SAIL continues to dispatch one `(query, operator)` per idle worker. It does
not create an outer query batch or change the worker/vLLM execution interface.

## Run

```bash
python -m mfe.scripts.experiment_baselines \
  --questions-file data/experiments_design7/mixed_medium_first100.jsonl \
  --scheduler rhsail \
  --output-max-tokens 2048 \
  --arrival-mode poisson-burst \
  --poisson-rate 0.15 \
  --arrival-batch-size 1
```

Equivalent environment selection:

```bash
export MFE_SCHEDULER=rhsail
```

## Objective

The rollout objective starts with the unfinished-request area, which aligns
with average completion time. It then adds normalized service-stretch and
inter-op-gap pressure. Hard thresholds prevent a started workflow from being
left without progress indefinitely.

Candidate construction deliberately includes different policies: largest
gap, largest stretch, unfinished commitment, completion opportunity,
critical-path/unlock value, SAIL affinity, HRRN-style admission, and shortest
predicted operator. The top candidates are evaluated with a short rollout.

## Main settings

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `MFE_RHSAIL_ACTIVE_DAG_FACTOR` | `3.0` | Active workflow limit as a multiple of worker count. |
| `MFE_RHSAIL_ACTIVE_DAG_LIMIT` | `0` | Explicit active limit; `0` uses the factor. |
| `MFE_RHSAIL_ADMISSION_MAX_WAIT` | `300` | Maximum waiting time before admission bypasses the active limit. |
| `MFE_RHSAIL_CANDIDATE_K` | `12` | Candidate budget after diverse construction. |
| `MFE_RHSAIL_ROLLOUT_HORIZON` | `3` | Number of shadow operator completions. |
| `MFE_RHSAIL_COMMITMENT_OPS` | `2` | Initial completed-op progress quantum. |
| `MFE_RHSAIL_SOFT_GAP_SECONDS` | `30` | Gap level where progressive penalty starts. |
| `MFE_RHSAIL_HARD_GAP_SECONDS` | `180` | Gap level that triggers emergency scheduling. |
| `MFE_RHSAIL_SOFT_STRETCH` | `2.0` | Projected service-stretch penalty threshold. |
| `MFE_RHSAIL_HARD_STRETCH` | `6.0` | Projected service-stretch emergency threshold. |
| `MFE_RHSAIL_USE_SAIL_GUIDANCE` | `1` | Enable cached SAIL placement/reuse guidance. |
| `MFE_RHSAIL_RUNTIME_EWMA_ALPHA` | `0.25` | Online runtime-model update rate. |

The scheduler snapshot records admission throttles, emergency decisions,
maximum observed gap/stretch, and runtime-model bucket count under
`scheduler_metrics.rhsail`.

## State affinity boundary

SAIL affinity is a soft runtime signal. RH-SAIL prefers the planned worker and
workers that executed parent/reuse-source operators, but it can choose another
idle worker when completion cost is lower. Real KV/prefix reuse still requires
backend support such as vLLM prefix caching; affinity alone does not move KV
state between workers.
