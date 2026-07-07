# Baseline DAG Experiments

This workflow builds fixed JSONL workloads for MFE scheduling experiments and
runs baseline schedulers before adding a custom paper algorithm.

## Build workloads

Download any datasets that are needed and available:

```bash
python -m mfe.scripts.download_datasets --datasets gsm8k math drop hotpotqa strategyqa mbpp --data-dir data --limit 200
```

Build deterministic JSONL workloads:

```bash
python -m mfe.scripts.build_experiment_datasets --data-dir data --output-dir data/experiments --size smoke --output-length medium
python -m mfe.scripts.build_experiment_datasets --data-dir data --output-dir data/experiments --size dev --output-length long
```

Each JSONL row contains `sample_id`, `dataset`, `dag_family`, `yaml`,
`question`, `answer`, estimated input length, output length regime, and source
metadata.

## Run baselines

Local scheduler smoke test without real accelerator devices:

```bash
python -m mfe.scripts.experiment_baselines \
  --questions-file data/experiments/mixed_medium_smoke.jsonl \
  --scheduler fcfs \
  --output-length medium \
  --test-worker \
  --worker-delay 0.01

python -m mfe.scripts.experiment_baselines \
  --questions-file data/experiments/mixed_medium_smoke.jsonl \
  --scheduler sjf \
  --output-length medium \
  --test-worker \
  --worker-delay 0.01
```

Unified entrypoint on target machines:

```bash
MFE_SCHEDULER=fcfs bash deploy/run_unified.sh lab-a800 \
  --model-path /data/mfe/models/Qwen3-0.6B \
  --device-ids 0,1,2,3 \
  --expected-device-count 4 \
  --questions-file data/experiments/mixed_long_dev.jsonl \
  --scheduler fcfs \
  --output-length long \
  --repeat 3 \
  --offline

MFE_SCHEDULER=sjf bash deploy/run_unified.sh company-ascend \
  --model-path /data/mfe/models/Qwen3-0.6B \
  --device-ids 0,1,2,3 \
  --expected-device-count 4 \
  --questions-file data/experiments/mixed_long_dev.jsonl \
  --scheduler sjf \
  --output-length long \
  --repeat 3 \
  --offline
```

## Metrics

The experiment runner writes per-run detail JSON, summary JSON, and summary CSV.
Summary metrics include:

- makespan, request throughput, input/output/total token throughput, and goodput;
- latency mean/P50/P95/P99, mean waiting time, service time, and scheduler overhead;
- critical-path estimate, DAG parallelism estimate, dependency stall time, and ready-queue stats;
- per-device busy percentage, per-device output token rate, load imbalance, and cross-device dependency count;
- per-DAG-family grouped counts, latency, and token throughput.

`--test-worker` uses synthetic token counts from `--output-length`; real vLLM
runs record token counts from vLLM/tokenizer outputs.
