#!/usr/bin/env bash
set -euo pipefail

requested_root="${1:?Usage: bash deploy/cancel_company_ascend_sweep.sh /absolute/path/to/run-root}"
RUN_ROOT="$(cd "$requested_root" 2>/dev/null && pwd)" || {
  echo "ERROR: run root does not exist: $requested_root" >&2
  exit 2
}

config="$RUN_ROOT/run_config.txt"
pid_file="$RUN_ROOT/runner.pid"
if [[ ! -f "$config" || ! -f "$pid_file" ]]; then
  echo "ERROR: run_config.txt or runner.pid is missing under: $RUN_ROOT" >&2
  exit 2
fi

request_count="$(sed -n 's/^request_count=//p' "$config")"
questions_file="$(sed -n 's/^questions_file=//p' "$config")"
if [[ "$request_count" != "1400" || "$questions_file" != *mixed_medium_first200.jsonl ]]; then
  echo "ERROR: refusing to cancel a run that is not the old first200/1400 batch" >&2
  echo "request_count=$request_count" >&2
  echo "questions_file=$questions_file" >&2
  exit 2
fi

runner_pid="$(tr -d '[:space:]' < "$pid_file")"
if [[ ! "$runner_pid" =~ ^[0-9]+$ ]]; then
  echo "ERROR: invalid runner PID: $runner_pid" >&2
  exit 2
fi
if ! kill -0 "$runner_pid" 2>/dev/null; then
  echo "Runner is already stopped: PID $runner_pid"
  if [[ -f "$RUN_ROOT/DONE" ]]; then
    echo "The old batch already has DONE; preserving its completion markers."
  else
    echo "No signal was needed. Inspect the existing FAILED/CANCELLED markers and logs."
  fi
  exit 0
fi

runner_cmd="$(tr '\0' ' ' < "/proc/$runner_pid/cmdline")"
if [[ "$runner_cmd" != *run_company_ascend_sweep.sh* ]]; then
  echo "ERROR: PID $runner_pid is not the company sweep runner" >&2
  echo "command=$runner_cmd" >&2
  exit 2
fi

collect_tree_postorder() {
  local parent="$1"
  local child
  while read -r child; do
    [[ -n "$child" ]] && collect_tree_postorder "$child"
  done < <(pgrep -P "$parent" 2>/dev/null || true)
  printf '%s\n' "$parent"
}

mapfile -t process_tree < <(collect_tree_postorder "$runner_pid")
pid_list="$(IFS=,; echo "${process_tree[*]}")"
declare -A process_start_times=()
for pid in "${process_tree[@]}"; do
  process_start_times["$pid"]="$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || true)"
done

echo "About to cancel the verified first200/1400 batch:"
echo "RUN_ROOT=$RUN_ROOT"
echo "request_count=$request_count"
echo "questions_file=$questions_file"
echo "runner_pid=$runner_pid"
ps -fp "$pid_list" || true
echo
read -r -p "Type CANCEL-1400 to terminate only this process tree: " confirmation
if [[ "$confirmation" != "CANCEL-1400" ]]; then
  echo "Cancelled by user; no signal was sent."
  exit 1
fi

kill -TERM "${process_tree[@]}" 2>/dev/null || true
for _ in $(seq 1 30); do
  alive=0
  for pid in "${process_tree[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive=1
      break
    fi
  done
  [[ "$alive" == "0" ]] && break
  sleep 1
done

remaining=()
for pid in "${process_tree[@]}"; do
  current_start="$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || true)"
  if [[ -n "$current_start" && "$current_start" == "${process_start_times[$pid]}" ]]; then
    remaining+=("$pid")
  fi
done
if [[ "${#remaining[@]}" -gt 0 ]]; then
  echo "Processes did not exit after 30 seconds; force-stopping the same verified tree: ${remaining[*]}"
  kill -KILL "${remaining[@]}" 2>/dev/null || true
fi

rm -f "$RUN_ROOT/DONE"
touch "$RUN_ROOT/CANCELLED" "$RUN_ROOT/FAILED"
printf '[%s] CANCELLED old first200/1400 batch by operator\n' "$(date --iso-8601=seconds)" \
  >> "$RUN_ROOT/runner.log"

echo "Cancellation complete. Old results were preserved under: $RUN_ROOT"
echo "Check that the five intended NPUs are free before starting the first50 run: npu-smi info"
