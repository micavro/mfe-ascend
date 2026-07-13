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
python -m mfe.scripts.build_experiment_datasets --data-dir data --output-dir data/experiments --size first100 --output-length medium
python -m mfe.scripts.build_experiment_datasets --data-dir data --output-dir data/experiments --size first200 --output-length medium
python -m mfe.scripts.build_experiment_datasets --data-dir data --output-dir data/experiments --size first500 --output-length medium
```

The prepared offline bundle for server upload is
`data/mfe_offline_medium_100_200_500.zip`. It contains the six parquet datasets
plus per-dataset and mixed JSONL workloads for 100, 200, and 500 examples per
dataset. The mixed workload sizes are 600, 1200, and 3000 rows.

The current DAG mapping is GSM8K -> chain/path, MATH -> reflect, DROP ->
parallel fork-join, HotpotQA -> parallel tree-reduce, StrategyQA -> debate, and
MBPP -> large mixed DAG.

If the Ascend machine cannot download public datasets, build a small built-in
workload instead:

```bash
python -m mfe.scripts.build_experiment_datasets \
  --builtin-tiny \
  --output-dir data/experiments \
  --output-length medium
```

This writes `data/experiments/mixed_medium_tiny.jsonl` and does not require any
external parquet files.

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
  --arrival-mode poisson-burst \
  --arrival-batch-size 4 \
  --poisson-rate 1.0 \
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
  --arrival-mode poisson-burst \
  --arrival-batch-size 4 \
  --poisson-rate 1.0 \
  --offline
```

Use `--scheduler sailp` for the SAI-LP scheduler. In `poisson-burst` mode,
`--arrival-batch-size` is only the number of independent queries submitted in one
client-side arrival burst. It does not enable runtime query batching; the runtime
still dispatches one `(query, operator)` at a time with `query_ids=[uid]`.
`--poisson-rate` is the burst arrival rate in bursts per second.

Use `--scheduler rhsail` for the completion-oriented online scheduler that
combines SAIL placement/state-affinity guidance with RHRS-style diverse
candidate rollout. See `docs/rhsail.md` for its progress and admission guards.

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

Every run also prints a compact copyable block to stdout and saves
`brief_summary.csv` plus `brief_summary.txt` in the output directory:

```text
MFE_BRIEF_RESULT_START
scheduler,output_length,repeat_index,count,completed,success_rate,makespan_s,total_tokens,output_tokens,total_tokens_per_s,output_tokens_per_s,req_per_s,avg_wait_s,p95_latency_s,ready_queue_peak,device_busy_pct,load_imbalance,parallelism_utilization
fcfs,medium,1,7,7,1.0,123.4567,89123,45678,721.63,370.04,0.0567,12.3456,40.1234,7,0:0.8123;1:0.7744;2:0.7655;3:0.8021,1.0612,0.7888
MFE_BRIEF_RESULT_END
```
